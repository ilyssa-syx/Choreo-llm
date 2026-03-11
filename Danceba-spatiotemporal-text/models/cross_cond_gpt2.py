"""
GPT model with sliding window and text temporal alignment fix
- Fixed variable name bug (self.cond_emb -> self.music_cond_emb)
- Added window_abs_start tracking for sliding window inference
- Fixed text_meta temporal alignment in attention mechanism
"""

import math
import warnings

import torch
import torch.nn as nn
from torch.nn import functional as F

from .mlp import GatedMLP
from .mamba_simple import Mamba
from .LPE import LPE_1

class CrossCondGPT2(nn.Module):
    """  Danceba Pipeline  """
    def __init__(self, config):
        super().__init__()
        self.gpt_base = CrossCondGPTBase(config.base)
        self.gpt_head = CrossCondGPTHead(config.head)
        self.block_size = config.block_size

    def get_block_size(self):
        return self.block_size

    def _resolve_text_modalities(self, text_upper, text_lower, text_torso, text_whole, text_simple_tag, music):
        """
        处理5个文本域（任意可为None），输出5个张量（None域用零张量占位）。
        保持5域结构固定，供后续拼接和mask逻辑使用。

        Args:
            text_upper/lower/torso/whole/simple_tag: (B, N, D) 或 (B, N, 1, D) 或 None
            music: 参考张量，仅用于在全部域为 None 时提供 device/dtype（当前路径已保守报错，
                   实际不会用到，保留参数以维持接口兼容性）

        Returns:
            text_upper, text_lower, text_torso, text_whole, text_simple_tag: 全是 Tensor (B, N, D)
            N: int, 文本段数
            text_dim: int, 文本特征维度
            domain_present: list[bool], 长度为5，True=真实域 / False=零占位域

        Raises:
            ValueError: 若所有5个文本域均为None（保守策略，要求至少保留1个域）
        """
        texts = [text_upper, text_lower, text_torso, text_whole, text_simple_tag]
        names = ['upper', 'lower', 'torso', 'whole', 'simple_tag']

        # 找到所有非None的域
        non_none = [(i, t) for i, t in enumerate(texts) if t is not None]

        # 保守策略：若全部为None，直接报错
        if len(non_none) == 0:
            raise ValueError(
                "[CrossCondGPT2._resolve_text_modalities] 所有5个text域均为None。"
                "当前实现要求至少保留1个文本域（保守策略）。"
                "若确需纯音乐条件，请明确实现 N=0 的情形。"
            )

        def _normalize_to_3d(t: torch.Tensor) -> torch.Tensor:
            """将 3D (B,N,D) 或 4D (B,N,1,D) 统一转为 3D (B,N,D)。
            注意：4D 时严格要求 K==1；若 K>1 则 masking/meta 逻辑会错位，故直接报错。
            """
            if t.ndim == 3:
                return t
            elif t.ndim == 4:
                B_, N_, K_, D_ = t.shape
                if K_ != 1:
                    raise ValueError(
                        f"[CrossCondGPT2._normalize_to_3d] 4D tensor 的 K 维必须为 1（当前 K={K_}）。"
                        " text_meta 按 segment 索引，K>1 会导致 token 数与 meta 长度不匹配。"
                    )
                return t.reshape(B_, N_, D_)
            else:
                raise ValueError(
                    f"[CrossCondGPT2._resolve_text_modalities] 期望 3D 或 4D tensor，"
                    f"实际收到 {t.ndim}D，shape={t.shape}。"
                )

        # 从第一个非None域推断参考形状，统一归一化到 (B, N, D)
        ref_idx, ref_raw = non_none[0]
        ref_tensor = _normalize_to_3d(ref_raw)
        B, N, D = ref_tensor.shape

        # 一致性检查：所有非None域归一化后的 (B, N, D) 必须相同
        for i, t_raw in non_none:
            t_norm = _normalize_to_3d(t_raw)
            if t_norm.shape != (B, N, D):
                raise ValueError(
                    f"[CrossCondGPT2._resolve_text_modalities] text_{names[i]} 归一化后 shape {t_norm.shape} "
                    f"与参考域 text_{names[ref_idx]} 归一化后 shape ({B},{N},{D}) 不一致。"
                )

        ref_device = ref_tensor.device
        ref_dtype  = ref_tensor.dtype

        # 构建 domain_present 列表，并用零张量填充 None 域
        domain_present = []
        resolved = []
        for i, t_raw in enumerate(texts):
            if t_raw is not None:
                domain_present.append(True)
                resolved.append(_normalize_to_3d(t_raw))
            else:
                domain_present.append(False)
                resolved.append(torch.zeros(B, N, D, device=ref_device, dtype=ref_dtype))

        return resolved[0], resolved[1], resolved[2], resolved[3], resolved[4], N, D, domain_present

    def sample(self, xs, cond, text_upper, text_lower, text_torso, text_whole, text_simple_tag, text_meta, shift=None):
        """
        自回归采样函数，支持text条件
        
        Args:
            xs: (x_up, x_down) - 初始motion序列
            cond: music条件
            text_upper, text_lower, text_torso, text_whole, text_simple_tag: 文本条件 (B, N, 1, dim)
            text_meta: 每个batch的文本时间信息列表
            shift: 滑动窗口步长
        """
        print("do sample!!!")
        
        block_size = self.get_block_size() - 1
        
        # 当序列变长超过block_size时，窗口一次向后滑动多少格
        if shift is not None:
            if shift <= 0:
                raise ValueError(f"[CrossCondGPT2.sample] shift must be >= 1, got {shift}")
            block_shift = min(shift, block_size)
        else:
            block_shift = block_size
        
        x_up, x_down = xs
        
        # 逐帧自回归生成
        for k in range(cond.size(1)):
            # === 1. 裁剪motion context（处理超长序列的滑动窗口）===
            current_len = x_up.size(1)  # 当前已生成的总长度
            
            if current_len <= block_size:
                x_cond_up = x_up
                x_cond_down = x_down
                # [新增] 当前窗口的绝对起始位置是 0
                window_abs_start = 0
            else:
                # 计算滑动窗口的 offset (从末尾往回数多少个)
                offset = (block_shift + (k - block_size - 1) % (block_size - block_shift + 1))
                window_start = -offset
                x_cond_up = x_up[:, window_start:]
                x_cond_down = x_down[:, window_start:]
                # [新增] 当前窗口的绝对起始位置 = 总长度 - offset
                window_abs_start = current_len - offset
            
            # === 2. 裁剪music condition（保持与motion同步）===
            if k < block_size:
                cond_input = cond[:, :k+1]
            else:
                window_start_idx = k - (block_shift + (k - block_size - 1) % (block_size - block_shift + 1)) + 1
                cond_input = cond[:, window_start_idx:k+1]
            
            # === 3. Text条件保持完整传递 ===
            # Text条件是全局的，不需要裁剪
            # Attention mask会根据当前帧位置(window_abs_start)和text_meta自动决定该attend到哪些text segments
            
            # === 4. Forward pass ===
            logits, _ = self.forward(
                idxs=(x_cond_up, x_cond_down),
                music=cond_input,
                text_upper=text_upper,
                text_lower=text_lower,
                text_torso=text_torso,
                text_whole=text_whole,
                text_simple_tag=text_simple_tag,
                text_meta=text_meta,
                window_abs_start=window_abs_start,
                targets=None,
            )
            
            logit_up, logit_down = logits
            
            # === 5. 取最后一个时间步的logits（对应当前生成的第k帧）===
            logit_up = logit_up[:, -1, :]
            logit_down = logit_down[:, -1, :]

            # === 6. 采样 ===
            # Top-k 采样（k>1）
            top_k = 1
            probs_up = F.softmax(logit_up, dim=-1)
            probs_down = F.softmax(logit_down, dim=-1)

            # 仅保留 Top-k 概率并归一化，再进行随机采样
            topk_up_vals, topk_up_idx = torch.topk(probs_up, k=top_k, dim=-1)
            topk_down_vals, topk_down_idx = torch.topk(probs_down, k=top_k, dim=-1)

            topk_up_probs = topk_up_vals / topk_up_vals.sum(dim=-1, keepdim=True)
            topk_down_probs = topk_down_vals / topk_down_vals.sum(dim=-1, keepdim=True)

            sampled_up = torch.multinomial(topk_up_probs, num_samples=1)
            sampled_down = torch.multinomial(topk_down_probs, num_samples=1)

            ix_up = topk_up_idx.gather(-1, sampled_up)
            ix_down = topk_down_idx.gather(-1, sampled_down)

            # === 7. 将新生成的token追加到序列中 ===
            x_up = torch.cat((x_up, ix_up), dim=1)
            x_down = torch.cat((x_down, ix_down), dim=1)

        return ([x_up], [x_down])


    def forward(self, idxs, music, text_upper, text_lower, text_torso, text_whole, text_simple_tag, text_meta, targets=None, window_abs_start=0):
        """
        Forward pass with window absolute start position
        
        Args:
            window_abs_start: 当前窗口在完整序列中的绝对起始帧位置（默认0表示从头开始）
        """
        
        idx_up, idx_down = idxs
        
        # 支持任意文本域为 None（ablation）：统一 resolve 为 tensor，None 域用零占位
        text_upper, text_lower, text_torso, text_whole, text_simple_tag, N, text_dim, domain_present = \
            self._resolve_text_modalities(text_upper, text_lower, text_torso, text_whole, text_simple_tag, music)
        T_motion = idx_up.size(1)  # Length of motion sequence

        feat = self.gpt_base(idx_up, idx_down, music, text_upper, text_lower, text_torso, text_whole, text_simple_tag, text_meta, window_abs_start=window_abs_start, domain_present=domain_present)
        logits_up, logits_down, loss_up, loss_down = self.gpt_head(feat, text_meta=text_meta, T_motion=T_motion, N=N, targets=targets, window_abs_start=window_abs_start, domain_present=domain_present)
        
        if loss_up is not None and loss_down is not None:
            loss = loss_up + loss_down
        else:
            loss = None

        return (logits_up, logits_down), loss


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block_Base(nn.Module):
    """ an Temporal-Gated Causal Attention (TGCA) block """

    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config.n_embd)
        self.norm2 = RMSNorm(config.n_embd)
        self.in_proj = nn.Linear(config.n_embd, config.n_embd)
        self.act_proj = nn.Linear(config.n_embd, config.n_embd)
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)

        self.attn = CausalCrossConditionalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, text_meta=None, T=None, N=None, window_abs_start=0, domain_present=None):
        """
        Args:
            window_abs_start: 窗口绝对起始帧位置
            domain_present: list[bool] 长度为5，传递给 attention 以过滤禁用域
        """
        shortcut = x
        x = self.norm1(x)
        # gate
        act_res = self.act(self.act_proj(x))
        
        x = self.in_proj(x)
        x = self.act(x)
        x = self.attn(x, text_meta=text_meta, T_motion=T, N=N, window_abs_start=window_abs_start, domain_present=domain_present)
        
        x = self.out_proj(x * act_res)
        x = shortcut + x

        x = x + self.mlp(self.norm2(x))
        return x


class Block_Head(nn.Module):
    """
    Hybrid Architecture: Mamba for Motion + GatedMLP for Text
    - Motion: 3 independent Mamba blocks (Music, Up, Down)
    - Text: GatedMLP for all 5 text modalities (Upper, Lower, Torso, Whole, SimpleTag)
    - Text uses lightweight GatedMLP instead of Mamba for efficiency
    - Each modality has independent RMSNorm for feature adaptation
    """

    def __init__(self, config):
        super().__init__()
        
        # === 7 Independent RMSNorm layers ===
        self.norm_music = RMSNorm(config.n_embd)
        self.norm_up = RMSNorm(config.n_embd)
        self.norm_down = RMSNorm(config.n_embd)
        self.norm_text_upper = RMSNorm(config.n_embd)
        self.norm_text_lower = RMSNorm(config.n_embd)
        self.norm_text_torso = RMSNorm(config.n_embd)
        self.norm_text_whole = RMSNorm(config.n_embd)
        self.norm_text_simple_tag = RMSNorm(config.n_embd)
        
        # === 3 Motion Mamba blocks ===
        self.mamba_music = Mamba(d_model=config.n_embd, d_state=128, d_conv=4, expand=4)
        self.mamba_up = Mamba(d_model=config.n_embd, d_state=128, d_conv=4, expand=4)
        self.mamba_down = Mamba(d_model=config.n_embd, d_state=128, d_conv=4, expand=4)
        
        # === GatedMLP for Text modalities ===
        self.text_mlp = GatedMLP(config.n_embd, config.n_embd * 4, config.n_embd)

    def forward(self, x, T_motion, text_meta):
        """
        Parallel processing of 8 modality streams
        
        Args:
            x: (B, Total_Len, D) where Total_Len = 3*T_motion + 5*N
            T_motion: Length of motion sequence
            text_meta: Metadata (not used in this simplified version)
        """
        t = T_motion
        B, Total_Len, D = x.shape
        
        # Calculate N and sequence boundaries
        text_total_len = Total_Len - 3 * t
        assert text_total_len % 5 == 0, (
            f"[Block_Head] text_total_len={text_total_len} 不能被 5 整除，"
            f"Total_Len={Total_Len}, T_motion={t}。请检查上游序列拼接。"
        )
        N = text_total_len // 5
        text_segment_len = N
        motion_len = 3 * t
        
        # === 1. Slice input into 8 independent streams ===
        music = x[:, :t, :]
        up = x[:, t:2*t, :]
        down = x[:, 2*t:motion_len, :]
        text_upper = x[:, motion_len:motion_len + text_segment_len, :]
        text_lower = x[:, motion_len + text_segment_len:motion_len + 2*text_segment_len, :]
        text_torso = x[:, motion_len + 2*text_segment_len:motion_len + 3*text_segment_len, :]
        text_whole = x[:, motion_len + 3*text_segment_len:motion_len + 4*text_segment_len, :]
        text_simple_tag = x[:, motion_len + 4*text_segment_len:motion_len + 5*text_segment_len, :]
        
        # === 2. Parallel processing ===
        # Motion: 3 independent Mamba blocks
        music = music + self.mamba_music(self.norm_music(music))
        up = up + self.mamba_up(self.norm_up(up))
        down = down + self.mamba_down(self.norm_down(down))
        
        # Text: Shared GatedMLP for all 5 modalities (lightweight and efficient!)
        # Each modality still has independent normalization for feature adaptation
        text_upper = text_upper + self.text_mlp(self.norm_text_upper(text_upper))
        text_lower = text_lower + self.text_mlp(self.norm_text_lower(text_lower))
        text_torso = text_torso + self.text_mlp(self.norm_text_torso(text_torso))
        text_whole = text_whole + self.text_mlp(self.norm_text_whole(text_whole))
        text_simple_tag = text_simple_tag + self.text_mlp(self.norm_text_simple_tag(text_simple_tag))
        
        # === 3. Reassemble sequence ===
        x = torch.cat([music, up, down, text_upper, text_lower, text_torso, text_whole, text_simple_tag], dim=1)
        
        return x


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class SMR(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, linear=False):
        super(SMR, self).__init__()
        self.conv = nn.Conv1d(in_features, out_features, kernel_size, stride=1)
        self.use_linear = linear
        if linear:
            self.linear = nn.Linear(in_features, out_features)
        self.pad = (kernel_size - 1, 0)
    
    def forward(self, x):
        # Input shape: (B, H, L)
        # Output shape: (B, H, L)
        if self.use_linear:
            factor = self.linear(self.conv(F.pad(x, self.pad, mode='constant', value=0.0)).transpose(1, 2)).transpose(1, 2)
        else:
            factor = self.conv(F.pad(x, self.pad, mode='constant', value=0.0))
        return torch.sigmoid(factor) * x


class CausalCrossConditionalSelfAttention(nn.Module):
    """
    优化版本的多头掩码自注意力层
    - 使用向量化操作替代Python循环，大幅提升GPU性能
    - 修复fallback逻辑，防止在包含文本特征时出现错误
    - 修复滑动窗口推理时的文本时间对齐问题 ⭐ NEW
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head

        # ── text_mask_mode ──────────────────────────────────────────────────────
        # Supported canonical modes:
        #   full        – body mask + temporal mask (default, matches original behaviour)
        #   time_only   – no body mask, keep temporal mask  (alias: no_body_mask)
        #   part_only   – no temporal mask, keep body mask  (alias: no_temporal_mask)
        #   none        – both off (music->text still forbidden) (alias: no_mask)
        _ALIAS_MAP = {
            "full":             "full",
            "time_only":        "time_only",
            "part_only":        "part_only",
            "none":             "none",
            "no_body_mask":     "time_only",
            "no_temporal_mask": "part_only",
            "no_mask":          "none",
        }
        raw_mode = getattr(config, "text_mask_mode", "full")
        canonical = _ALIAS_MAP.get(raw_mode, None)
        if canonical is None:
            import warnings
            warnings.warn(
                f"[CausalCrossConditionalSelfAttention] Unknown text_mask_mode '{raw_mode}', "
                "falling back to 'full'.",
                UserWarning,
            )
            canonical = "full"
        self.text_mask_mode = canonical
        # text_to_motion_attn: True = text CAN attend to motion (Danceba bidirectional style)
        # False = text BLOCKED from motion (default for this repo)
        self.text_to_motion_attn = getattr(config, 'text_to_motion_attn', False)

    def forward(self, x, text_meta=None, T_motion=None, N=None, window_abs_start=0, domain_present=None):
        """
        Args:
            x: Input tensor
            text_meta: Text temporal metadata
            T_motion: Motion sequence length
            N: Number of text segments
            window_abs_start: 当前窗口在完整序列中的绝对起始帧位置 ⭐ KEY FIX
            domain_present: list[bool] 长度为5，True=允许该文本域被 attend / False=全局禁用该域
        """
        B, Total_Len, C = x.size()  # Total_Len = 3*T_motion + 5*N*1 (music, up, down, 5 text features)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, Total_Len, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, Total_Len, hs)
        q = self.query(x).view(B, Total_Len, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, Total_Len, hs)
        v = self.value(x).view(B, Total_Len, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, Total_Len, hs)
        
        # causal self-attention; Self-attend: (B, nh, Total_Len, hs) x (B, nh, hs, Total_Len) -> (B, nh, Total_Len, Total_Len)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # Construct spatio-temporal mask
        if T_motion is not None and N is not None:
            # ── 主分支：T_motion 和 N 已知，可正确构建完整 mask ──────────────
            T_motion = int(T_motion)
            N = int(N)

            # 长度一致性断言：提前暴露上游拼接错误
            expected_total_len = 3 * T_motion + 5 * N
            if Total_Len != expected_total_len:
                raise ValueError(
                    f"[CausalCrossConditionalSelfAttention] Total_Len 不匹配: "
                    f"got {Total_Len}, expected 3*T_motion+5*N={expected_total_len} "
                    f"(T_motion={T_motion}, N={N})"
                )

            # Create st_mask: shape (B, 1, Total_Len, Total_Len)
            # True = Forbid (will be filled with -inf), False = Allow
            st_mask = torch.zeros(B, 1, Total_Len, Total_Len, device=x.device, dtype=torch.bool)

            # Logic 1: Motion-to-Motion - Vectorized Causal Mask for 3-stream (music, up, down)
            motion_len = 3 * T_motion

            # 序列结构：[music_0...music_{T-1}, up_0...up_{T-1}, down_0...down_{T-1}]
            # 索引范围：music[0:T], up[T:2T], down[2T:3T]

            # 初始化：motion-to-motion 默认全部禁止
            st_mask[:, :, :motion_len, :motion_len] = True

            # 🚀 向量化实现：使用torch.tril一次性构建所有causal masks
            causal_template = torch.tril(torch.ones(T_motion, T_motion, device=x.device, dtype=torch.bool))
            strictly_lower  = torch.tril(torch.ones(T_motion, T_motion, device=x.device, dtype=torch.bool), diagonal=-1)

            # === Music stream 的因果关系 ===
            st_mask[:, :, 0:T_motion, 0:T_motion] = ~causal_template

            # === Up stream 的因果关系 ===
            st_mask[:, :, T_motion:2*T_motion, 0:T_motion]           = ~causal_template   # up -> music
            st_mask[:, :, T_motion:2*T_motion, T_motion:2*T_motion]  = ~strictly_lower    # up -> up
            st_mask[:, :, T_motion:2*T_motion, 2*T_motion:3*T_motion]= ~strictly_lower    # up -> down

            # === Down stream 的因果关系 ===
            st_mask[:, :, 2*T_motion:3*T_motion, 0:T_motion]          = ~causal_template  # down -> music
            st_mask[:, :, 2*T_motion:3*T_motion, T_motion:2*T_motion] = ~causal_template  # down -> up (incl. same t)
            st_mask[:, :, 2*T_motion:3*T_motion, 2*T_motion:3*T_motion] = ~strictly_lower # down -> down

            # Logic 2: Motion-to-Text - Default ALL FORBIDDEN
            st_mask[:, :, :motion_len, motion_len:] = True

            # ⚡ Fix #1: text->motion controlled by text_to_motion_attn flag
            # False (default) = text BLOCKED from motion (no information leakage)
            # True = text CAN attend to motion (Danceba bidirectional style)
            if not self.text_to_motion_attn:
                st_mask[:, :, motion_len:, :motion_len] = True

            # ── Text feature region indices ───────────────────────────────────
            text_start            = motion_len
            text_upper_start      = text_start
            text_lower_start      = text_start + N
            text_torso_start      = text_start + 2 * N
            text_whole_start      = text_start + 3 * N
            text_simple_tag_start = text_start + 4 * N

            # ── Domain-presence switches (respected in ALL modes) ─────────────
            upper_on      = domain_present is None or domain_present[0]
            lower_on      = domain_present is None or domain_present[1]
            torso_on      = domain_present is None or domain_present[2]
            whole_on      = domain_present is None or domain_present[3]
            simple_tag_on = domain_present is None or domain_present[4]

            text_mask_mode = getattr(self, 'text_mask_mode', 'full')

            if text_mask_mode in ('part_only', 'none'):
                # ⚡ Fix #2: part_only/none 不依赖 text_meta，直接对全 N token 开放
                up_all   = slice(T_motion,     2 * T_motion)
                down_all = slice(2 * T_motion, 3 * T_motion)

                if text_mask_mode == 'part_only':
                    # Body partition only – no time restriction
                    # up -> upper / torso / whole / simple_tag
                    if upper_on:      st_mask[:, :, up_all, text_upper_start      :text_upper_start + N     ] = False
                    if torso_on:      st_mask[:, :, up_all, text_torso_start      :text_torso_start + N     ] = False
                    if whole_on:      st_mask[:, :, up_all, text_whole_start      :text_whole_start + N     ] = False
                    if simple_tag_on: st_mask[:, :, up_all, text_simple_tag_start :text_simple_tag_start + N] = False
                    # down -> lower / torso / whole / simple_tag
                    if lower_on:      st_mask[:, :, down_all, text_lower_start     :text_lower_start + N     ] = False
                    if torso_on:      st_mask[:, :, down_all, text_torso_start     :text_torso_start + N     ] = False
                    if whole_on:      st_mask[:, :, down_all, text_whole_start     :text_whole_start + N     ] = False
                    if simple_tag_on: st_mask[:, :, down_all, text_simple_tag_start:text_simple_tag_start + N] = False

                else:  # none – no body, no temporal; music->text still forbidden
                    for q_all in (up_all, down_all):
                        if upper_on:      st_mask[:, :, q_all, text_upper_start      :text_upper_start + N     ] = False
                        if lower_on:      st_mask[:, :, q_all, text_lower_start      :text_lower_start + N     ] = False
                        if torso_on:      st_mask[:, :, q_all, text_torso_start      :text_torso_start + N     ] = False
                        if whole_on:      st_mask[:, :, q_all, text_whole_start      :text_whole_start + N     ] = False
                        if simple_tag_on: st_mask[:, :, q_all, text_simple_tag_start :text_simple_tag_start + N] = False

            elif text_meta is not None:
                # batch 结构校验
                if not isinstance(text_meta, (list, tuple)) or len(text_meta) != B:
                    raise ValueError(
                        f"[CausalCrossConditionalSelfAttention] text_meta 必须是长度为 B={B} 的 list/tuple，"
                        f"got type={type(text_meta)}, "
                        f"len={len(text_meta) if hasattr(text_meta, '__len__') else 'N/A'}"
                    )
                # ── 'full' or 'time_only': temporal loop over text_meta ───────
                # ⭐ 核心对齐：text_meta 帧号 // 8 后减去 window_abs_start 得到窗口相对位置
                for b in range(B):
                    if text_meta[b] is None or len(text_meta[b]) == 0:
                        continue

                    for i, meta in enumerate(text_meta[b]):
                        if i >= N:  # Safety check
                            break

                        abs_start = meta['start_frame'] // 8
                        abs_end   = meta['end_frame']   // 8
                        rel_start = abs_start - window_abs_start
                        rel_end   = abs_end   - window_abs_start

                        seg_start = i
                        seg_end   = i + 1

                        frame_start = max(0, rel_start)
                        frame_end   = min(T_motion, rel_end)

                        if frame_end > frame_start:
                            up_slice   = slice(T_motion     + frame_start, T_motion     + frame_end)
                            down_slice = slice(2 * T_motion + frame_start, 2 * T_motion + frame_end)

                            text_upper_slice      = slice(text_upper_start      + seg_start, text_upper_start      + seg_end)
                            text_lower_slice      = slice(text_lower_start      + seg_start, text_lower_start      + seg_end)
                            text_torso_slice      = slice(text_torso_start      + seg_start, text_torso_start      + seg_end)
                            text_whole_slice      = slice(text_whole_start      + seg_start, text_whole_start      + seg_end)
                            text_simple_tag_slice = slice(text_simple_tag_start + seg_start, text_simple_tag_start + seg_end)

                            if text_mask_mode == 'full':
                                # Body + temporal: up sees upper/torso/whole/simple_tag
                                if upper_on:      st_mask[b, 0, up_slice, text_upper_slice]      = False
                                if torso_on:      st_mask[b, 0, up_slice, text_torso_slice]      = False
                                if whole_on:      st_mask[b, 0, up_slice, text_whole_slice]      = False
                                if simple_tag_on: st_mask[b, 0, up_slice, text_simple_tag_slice] = False
                                # down sees lower/torso/whole/simple_tag
                                if lower_on:      st_mask[b, 0, down_slice, text_lower_slice]      = False
                                if torso_on:      st_mask[b, 0, down_slice, text_torso_slice]      = False
                                if whole_on:      st_mask[b, 0, down_slice, text_whole_slice]      = False
                                if simple_tag_on: st_mask[b, 0, down_slice, text_simple_tag_slice] = False

                            else:  # time_only – temporal on, no body: both up/down see all enabled domains
                                for q_slice in (up_slice, down_slice):
                                    if upper_on:      st_mask[b, 0, q_slice, text_upper_slice]      = False
                                    if lower_on:      st_mask[b, 0, q_slice, text_lower_slice]      = False
                                    if torso_on:      st_mask[b, 0, q_slice, text_torso_slice]      = False
                                    if whole_on:      st_mask[b, 0, q_slice, text_whole_slice]      = False
                                    if simple_tag_on: st_mask[b, 0, q_slice, text_simple_tag_slice] = False
            # else: full/time_only 但 text_meta 为 None → text 保持全禁（安全，不泄漏）
            # ⚠️ 若预期使用 text 条件但实验中 text_meta 为 None，mask 静默退化为 no-text
            else:
                warnings.warn(
                    f"[CausalCrossConditionalSelfAttention] text_mask_mode={getattr(self, 'text_mask_mode', 'full')!r} "
                    "需要 text_meta 做时间门控，但收到 text_meta=None。"
                    "motion query 对 text 的 attention 将保持全禁，等效于不使用 text 条件。"
                    "请检查 dataloader 或采样调用是否正确传入 text_meta。",
                    UserWarning,
                    stacklevel=4,
                )

            # ── Method B: 禁用域列屏蔽 ───────────────────────────────────────────
            # 将所有禁用文本域对应的列（key 位置）对全部 query（含 text->text）标为 forbidden。
            # 即使禁用域 token 在 MLP/bias 后变为非零，任何 token 也无法 attend 到它们，
            # 从而彻底截断"禁用域经 text->text attention 污染启用域"这条间接泄漏路径。
            if domain_present is not None:
                _domain_starts = [
                    text_upper_start,
                    text_lower_start,
                    text_torso_start,
                    text_whole_start,
                    text_simple_tag_start,
                ]
                for _dom_idx, (_present, _start) in enumerate(zip(domain_present, _domain_starts)):
                    if not _present:
                        st_mask[:, :, :, _start:_start + N] = True

            # Apply mask to attention scores
            att = att.masked_fill(st_mask, float('-inf'))
        else:
            # ── Fallback：T_motion 或 N 未知，只能做纯 motion causal mask ──
            if T_motion is not None:
                motion_len = 3 * int(T_motion)

                if Total_Len == motion_len:
                    # 纯 motion 数据：3-stream causal mask
                    st_mask = torch.zeros(1, 1, Total_Len, Total_Len, device=x.device, dtype=torch.bool)
                    st_mask[:, :, :, :] = True

                    T = int(T_motion)
                    causal_template = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
                    strictly_lower  = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=-1)

                    st_mask[:, :, 0:T, 0:T]       = ~causal_template
                    st_mask[:, :, T:2*T, 0:T]     = ~causal_template
                    st_mask[:, :, T:2*T, T:2*T]   = ~strictly_lower
                    st_mask[:, :, T:2*T, 2*T:3*T] = ~strictly_lower
                    st_mask[:, :, 2*T:3*T, 0:T]   = ~causal_template
                    st_mask[:, :, 2*T:3*T, T:2*T] = ~causal_template
                    st_mask[:, :, 2*T:3*T, 2*T:3*T] = ~strictly_lower

                    att = att.masked_fill(st_mask, float('-inf'))
                elif Total_Len > motion_len:
                    # 含文本特征但缺乏 text_meta & N：motion 正常 causal，
                    # text->motion 禁止，text->text 放开（⚡ Fix #3: 防止 all-inf 导致 NaN）
                    st_mask = torch.zeros(1, 1, Total_Len, Total_Len, device=x.device, dtype=torch.bool)
                    st_mask[:, :, :, :] = True

                    T = int(T_motion)
                    causal_template = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
                    strictly_lower  = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=-1)

                    st_mask[:, :, 0:T, 0:T]       = ~causal_template
                    st_mask[:, :, T:2*T, 0:T]     = ~causal_template
                    st_mask[:, :, T:2*T, T:2*T]   = ~strictly_lower
                    st_mask[:, :, T:2*T, 2*T:3*T] = ~strictly_lower
                    st_mask[:, :, 2*T:3*T, 0:T]   = ~causal_template
                    st_mask[:, :, 2*T:3*T, T:2*T] = ~causal_template
                    st_mask[:, :, 2*T:3*T, 2*T:3*T] = ~strictly_lower

                    # text->text 放开，避免 softmax(all -inf) = NaN
                    st_mask[:, :, motion_len:, motion_len:] = False
                    # text->motion 保持禁止（已是 True，无需重写）

                    att = att.masked_fill(st_mask, float('-inf'))
                else:
                    raise ValueError(f"Total_Len ({Total_Len}) < expected motion_len ({motion_len})")
            else:
                # 完全不知道结构：假设等分 3 段 motion
                t = Total_Len // 3
                if Total_Len == t * 3:
                    st_mask = torch.zeros(1, 1, Total_Len, Total_Len, device=x.device, dtype=torch.bool)
                    st_mask[:, :, :, :] = True

                    causal_template = torch.tril(torch.ones(t, t, device=x.device, dtype=torch.bool))
                    strictly_lower  = torch.tril(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=-1)

                    st_mask[:, :, 0:t, 0:t]       = ~causal_template
                    st_mask[:, :, t:2*t, 0:t]     = ~causal_template
                    st_mask[:, :, t:2*t, t:2*t]   = ~strictly_lower
                    st_mask[:, :, t:2*t, 2*t:3*t] = ~strictly_lower
                    st_mask[:, :, 2*t:3*t, 0:t]   = ~causal_template
                    st_mask[:, :, 2*t:3*t, t:2*t] = ~causal_template
                    st_mask[:, :, 2*t:3*t, 2*t:3*t] = ~strictly_lower

                    att = att.masked_fill(st_mask, float('-inf'))
                else:
                    raise ValueError(
                        f"Cannot infer structure: Total_Len={Total_Len}, T_motion={T_motion}, N={N}. "
                        "Please provide T_motion and N for sequences with text features."
                    )
        
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, Total_Len, Total_Len) x (B, nh, Total_Len, hs) -> (B, nh, Total_Len, hs)
        y = y.transpose(1, 2).contiguous().view(B, Total_Len, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class CrossCondGPTBase(nn.Module):
    """  the Global Beat Attention via Temporal Gating in Sec 3.3 """

    def __init__(self, config):
        super().__init__()
        self.tok_emb_up = nn.Embedding(config.vocab_size_up, config.n_embd)
        self.tok_emb_down = nn.Embedding(config.vocab_size_down, config.n_embd)
        """  Phase-Based Rhythm Feature Extraction in Sec 3.2  """
        self.pos_emb = LPE_1(config)
        self.position_scale = nn.Parameter(torch.tensor(1e-6))
        self.music_cond_emb = nn.Linear(config.n_music, config.n_embd)
        self.text_cond_emb = nn.Linear(config.n_text, config.n_embd)
        self.text_modality_emb = nn.Parameter(torch.randn(1, 1, config.n_embd) * 0.02)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.blocks = nn.Sequential(*[Block_Base(config) for _ in range(config.n_layer)])
        self.block_size = config.block_size
        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        
        # 【修复】加入 Conv1d 支持 Mamba
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d)
        
        # 【修复】显式加入自定义 RMSNorm 类
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, RMSNorm)
        
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                # 【修复】优先捕获独立的 Parameter (text_modality_emb, position_scale, learnable_padding_token)
                # 只要名字里包含这些特定的词，直接进入 no_decay (不减肥)
                if 'text_modality_emb' in pn or 'position_scale' in pn or 'learnable_padding_token' in pn:
                    no_decay.add(fpn)
                    continue  # 处理完直接跳过，防止后面逻辑干扰

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        
        # 如果还是报错，打印出来看看到底是谁漏了
        missing_keys = param_dict.keys() - union_params
        assert len(missing_keys) == 0, f"Critical Error: parameters {str(missing_keys)} were not separated into either decay/no_decay set!"

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx_up, idx_down, music, text_upper, text_lower, text_torso, text_whole, text_simple_tag, text_meta, window_abs_start=0, domain_present=None):
        """
        Args:
            window_abs_start: 当前窗口在完整序列中的绝对起始帧位置
            domain_present: list[bool] 长度为5，True=真实域 / False=零占位域（ablation掉的域）。
                            若为None则默认全部域均存在（向后兼容）。
        """
        # 防御式断言：所有 text tensor 必须是 Tensor（不允许 None 传入 base）
        for _name, _tensor in [('text_upper', text_upper), ('text_lower', text_lower),
                                ('text_torso', text_torso), ('text_whole', text_whole),
                                ('text_simple_tag', text_simple_tag)]:
            assert isinstance(_tensor, torch.Tensor), (
                f"[CrossCondGPTBase] {_name} must be a Tensor (got {type(_tensor)}). "
                "Please ensure _resolve_text_modalities() is called before passing to gpt_base."
            )
        b, t = idx_up.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        b, t = idx_down.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        token_embeddings_up = self.tok_emb_up(idx_up)  # each index maps to a (learnable) vector
        token_embeddings_down = self.tok_emb_down(idx_down)  # each index maps to a (learnable) vector
        
        # ✅ 修正Bug：使用 self.music_cond_emb 而不是 self.cond_emb
        token_embeddings = torch.cat([self.music_cond_emb(music), token_embeddings_up, token_embeddings_down], dim=1)
        
        position_embeddings = self.pos_emb(music)
        # 注意这里是3个并列的lpe预测然后concatenate。
        pos_size = token_embeddings.shape[1]
        position_embeddings = position_embeddings[:, :pos_size, :]
        position_embeddings = self.position_scale * position_embeddings
        token_embeddings = token_embeddings + position_embeddings
        
        # deal with text modality
        # reshape text conditions from (32, N, 1, 512) to (32, N*1, 512)
        N = text_upper.size(1)
        text_upper = text_upper.view(b, -1, text_upper.size(-1))  # (b, len, dim)
        text_lower = text_lower.view(b, -1, text_lower.size(-1))  # (b, len, dim)
        text_torso = text_torso.view(b, -1, text_torso.size(-1))  # (b, len, dim)
        text_whole = text_whole.view(b, -1, text_whole.size(-1))  # (b, len, dim)
        text_simple_tag = text_simple_tag.view(b, -1, text_simple_tag.size(-1))  # (b, len, dim)
        text_cond = torch.cat([text_upper, text_lower, text_torso, text_whole, text_simple_tag], dim=1)  # (b, total_len, dim)
        text_cond_emb = self.text_cond_emb(text_cond)  # (b, total_len, n_embd)
        text_modality_emb = self.text_modality_emb.repeat(1, text_cond_emb.size(1), 1)  # (1, total_len, n_embd)
        text_cond_emb = text_cond_emb + text_modality_emb

        # === 严格语义移除：对禁用域施加 text_valid_mask ===
        # 即使是零向量，经过 Linear bias + text_modality_emb 后也可能有固定信号。
        # text_valid_mask 彻底归零禁用域，保证其不向模型传递任何语义信息。
        # 这是全局 ablation 实验正确性的关键步骤。
        if domain_present is not None and not all(domain_present):
            # 每个域有 N tokens，5 个域共 5*N tokens
            # domain_present[i]=True -> 1.0（保留）; False -> 0.0（彻底归零）
            mask_values = []
            for present in domain_present:
                mask_values.extend([1.0 if present else 0.0] * N)
            text_valid_mask = torch.tensor(
                mask_values, device=text_cond_emb.device, dtype=text_cond_emb.dtype
            ).view(1, 5 * N, 1)
            text_cond_emb = text_cond_emb * text_valid_mask

        final_embedding = torch.cat([token_embeddings, text_cond_emb], dim=1)
        x = self.drop(final_embedding)

        T_motion = t  # Length of each motion segment (music/up/down)

        # ── Method A: 预先构建 re-zero 序列掩码 ─────────────────────────────────
        # 每个 Block_Base 之后将禁用文本域的 token 重新归零，防止 MLP/bias 在层间
        # 残差通路中累积"假语义"，与 Method B（attention 列屏蔽）形成双重保障。
        if domain_present is not None and not all(domain_present):
            _seq_len_base = 3 * T_motion + 5 * N
            _rz_vals_base = [1.0] * (3 * T_motion)  # motion tokens 始终保留
            for _present in domain_present:
                _rz_vals_base.extend([1.0 if _present else 0.0] * N)
            _re_zero_mask_base = torch.tensor(
                _rz_vals_base, device=x.device, dtype=x.dtype
            ).view(1, _seq_len_base, 1)
        else:
            _re_zero_mask_base = None

        for block in self.blocks:
            x = block(x, text_meta=text_meta, T=T_motion, N=N, window_abs_start=window_abs_start, domain_present=domain_present)
            # re-zero：确保禁用域 token 不在残差路径中积累非零值
            if _re_zero_mask_base is not None:
                x = x * _re_zero_mask_base

        return x


class CrossCondGPTHead(nn.Module):
    """  the Mamba-Based Parallel Motion Modeling in Sec 3.4 """

    def __init__(self, config):
        super().__init__()

        self.blocks = nn.Sequential(*[Block_Head(config) for _ in range(config.n_layer)])
        self.block_base = Block_Base(config)
        # decoder head
        self.RMS_f = RMSNorm(config.n_embd)
        self.block_size = config.block_size
        self.head_up = nn.Linear(config.n_embd, config.vocab_size_up, bias=False)
        self.head_down = nn.Linear(config.n_embd, config.vocab_size_down, bias=False)
        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        
        # 【修复】加入 Conv1d 支持 Mamba
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d)
        
        # 【修复】显式加入自定义 RMSNorm 类
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, RMSNorm)
        
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                # 【修复】优先捕获独立的 Parameter (text_modality_emb, position_scale, learnable_padding_token)
                # 只要名字里包含这些特定的词，直接进入 no_decay (不减肥)
                if 'text_modality_emb' in pn or 'position_scale' in pn or 'learnable_padding_token' in pn:
                    no_decay.add(fpn)
                    continue  # 处理完直接跳过，防止后面逻辑干扰

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        
        # 如果还是报错，打印出来看看到底是谁漏了
        missing_keys = param_dict.keys() - union_params
        assert len(missing_keys) == 0, f"Critical Error: parameters {str(missing_keys)} were not separated into either decay/no_decay set!"

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, x, text_meta=None, T_motion=None, N=None, targets=None, window_abs_start=0, domain_present=None):
        """
        Args:
            window_abs_start: 窗口绝对起始帧位置（传递给最后的block_base）
            domain_present: list[bool] 长度为5，传递给 block_base 的 attention
        """
        # ── Method A: 预先构建 re-zero 序列掩码（Head 同 Base 逻辑）───────────
        if domain_present is not None and not all(domain_present) and T_motion is not None and N is not None:
            _seq_len_head = 3 * T_motion + 5 * N
            _rz_vals_head = [1.0] * (3 * T_motion)
            for _present in domain_present:
                _rz_vals_head.extend([1.0 if _present else 0.0] * N)
            _re_zero_mask_head = torch.tensor(
                _rz_vals_head, device=x.device, dtype=x.dtype
            ).view(1, _seq_len_head, 1)
        else:
            _re_zero_mask_head = None

        for block in self.blocks:
            x = block(x, T_motion=T_motion, text_meta=text_meta)
            # re-zero：Block_Head 里的 Mamba/GatedMLP 也会给禁用域残差累积非零值
            if _re_zero_mask_head is not None:
                x = x * _re_zero_mask_head
        
        x = self.block_base(x, text_meta=text_meta, T=T_motion, N=N, window_abs_start=window_abs_start, domain_present=domain_present)
        # re-zero：block_base（最后的 TGCA 块）结束后再做一次，确保读取 logits 前干净
        if _re_zero_mask_head is not None:
            x = x * _re_zero_mask_head
        
        x = self.RMS_f(x)
        logits_up = self.head_up(x[:, T_motion:T_motion*2, :])
        logits_down = self.head_down(x[:, T_motion*2:T_motion*3, :])  # down half

        loss_up, loss_down = None, None

        if targets is not None:
            targets_up, targets_down = targets

            loss_up = F.cross_entropy(logits_up.view(-1, logits_up.size(-1)), targets_up.view(-1))
            loss_down = F.cross_entropy(logits_down.view(-1, logits_down.size(-1)), targets_down.view(-1))

        return logits_up, logits_down, loss_up, loss_down