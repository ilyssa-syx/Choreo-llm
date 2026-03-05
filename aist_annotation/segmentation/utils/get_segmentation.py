import numpy as np

def get_block_starts(arr):
    """
    识别所有连续block的起始位置。
    如果有一个block的长度为1，那么将它与前面或后面的block进行合并。
    合并规则：
    1. 如果前面没有block，与后面合并。
    2. 如果后面没有block，与前面合并。
    3. 如果前后都有block，与更短的block合并。
    
    参数:
    arr: numpy数组，包含聚类标签
    
    返回:
    block_starts: 合并后每个block的起始索引列表
    """
    if len(arr) == 0:
        return []
    
    # --- 步骤 1: 找到所有原始 block 的起始位置 (与原函数相同) ---
    block_starts = [0]  # 第一个block从索引0开始
    
    # 找到所有值变化的位置
    for i in range(1, len(arr)):
        if arr[i] != arr[i-1]:
            block_starts.append(i)
    
    # --- 步骤 2: 处理长度为1的 block ---
    
    # 如果总共只有0或1个block，那么不可能有长度为1的block需要合并
    # (唯一的block长度为 len(arr)，除非 len(arr) == 1，
    # 此时合并也无意义)
    if len(block_starts) <= 1:
        return block_starts
        
    i = 0
    # 使用 while 循环，因为我们会在循环中修改 block_starts 列表
    while i < len(block_starts):
        
        # (a) 计算当前 block (i) 的长度
        current_start = block_starts[i]
        is_last_block = (i == len(block_starts) - 1)
        
        if is_last_block:
            # 这是最后一个 block
            len_current = len(arr) - current_start
        else:
            # 这不是最后一个 block
            len_current = block_starts[i+1] - current_start
            
        # (b) 检查长度是否为 1
        if len_current == 1:
            # (c) 应用合并规则
            
            # 规则 1: 如果是第一个 block (i=0)，与后面合并
            # (此时 is_last_block 必定为 False, 因为 len(block_starts) > 1)
            if i == 0:
                # 合并方式：移除 *下一个* block 的 start (block_starts[i+1])
                # 这样，当前的 block[i] (即 [0]) 就自动“吞并”了下一个 block
                block_starts.pop(i + 1)
                # 不递增 i，因为 block_starts[0] 对应的 block 已经改变，
                # 我们需要重新检查这个新 block (它可能也需要合并)
            
            # 规则 2: 如果是最后一个 block (is_last_block=True)，与前面合并
            elif is_last_block:
                # 合并方式：移除 *当前* block 的 start (block_starts[i])
                # 这样，前一个 block 自动“吞并”了这最后一个元素
                block_starts.pop(i)
                # 不递增 i，循环将在下一次检查时 (i == len(block_starts)) 自动终止
            
            # 规则 3: 前后都有 block，与更短的合并
            else:
                # 计算前一个 block 的长度
                len_prev = block_starts[i] - block_starts[i-1]
                
                # 计算后一个 block 的长度
                is_next_block_the_last = (i + 1 == len(block_starts) - 1)
                if is_next_block_the_last:
                    len_next = len(arr) - block_starts[i+1]
                else:
                    len_next = block_starts[i+2] - block_starts[i+1]
                    
                # 比较并合并
                if len_prev <= len_next:
                    # 与前面合并：移除 *当前* block 的 start (block_starts[i])
                    block_starts.pop(i)
                    # 不递增 i，因为 block_starts[i] 现在是 *新* 的 block，
                    # 它也可能是长度为1，需要重新检查
                else:
                    # 与后面合并：移除 *下一个* block 的 start (block_starts[i+1])
                    block_starts.pop(i + 1)
                    # 不递增 i，因为 block_starts[i] 对应的 block 结尾变了，
                    # 它的长度也变了，需要重新检查
        
        else:
            # (d) 如果当前 block 长度不为 1，则继续检查下一个
            i += 1
    block_starts = np.array(block_starts)
    block_starts = np.round(block_starts * 150 / 18).astype(int).tolist()
    print(block_starts)  
    return block_starts



