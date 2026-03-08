#!/usr/bin/env python3
"""
测试域检查功能和modifier格式化功能的脚本
"""

# 模拟 check_response_domains 函数
def check_response_domains(response_text, required_domains=None):
    """
    检查Gemini回复是否包含所有必需的域（domain）
    
    Args:
        response_text: Gemini的回复文本
        required_domains: 必需的域列表，默认为 ['whole body', 'lower half body', 'upper half body', 'torso']
    
    Returns:
        (is_valid, missing_domains): is_valid为True表示所有域都存在，missing_domains为缺失的域列表
    """
    if required_domains is None:
        required_domains = ['whole body', 'lower half body', 'upper half body', 'torso']
    
    missing_domains = []
    
    for domain in required_domains:
        # 检查是否包含 **domain** 格式（不区分大小写）
        pattern = f"**{domain}**"
        if pattern.lower() not in response_text.lower():
            missing_domains.append(domain)
    
    is_valid = len(missing_domains) == 0
    return is_valid, missing_domains


# 模拟 format_modifier_text 函数
def format_modifier_text(modifiers):
    """
    将modifier列表格式化为交替的Pose和Trans格式
    格式：Pose 1, Trans 1, Pose 2, Trans 2, ..., Pose N
    Pose的数量比Trans多1个（最后总是以Pose结尾）
    
    Args:
        modifiers: modifier文本列表
    
    Returns:
        格式化后的文本
    """
    if not modifiers:
        return ""
    
    formatted_lines = []
    for i, modifier in enumerate(modifiers):
        # 偶数索引（0, 2, 4, ...）是Pose，奇数索引（1, 3, 5, ...）是Trans
        if i % 2 == 0:
            # Pose: 第 i//2 + 1 个
            pose_num = i // 2 + 1
            formatted_lines.append(f"* **Pose {pose_num}:** {modifier}")
        else:
            # Trans: 第 i//2 + 1 个
            trans_num = i // 2 + 1
            formatted_lines.append(f"* **Trans {trans_num}:** {modifier}")
    
    return "\n".join(formatted_lines)


# 测试用例
def test_domain_check():
    print("=" * 80)
    print("域检查功能测试")
    print("=" * 80)
    
    # 测试1: 完整的回复（包含所有4个域）
    complete_response = """* **whole body:** The dancer leans forward and hops on his right foot while repeatedly kicking his left leg backward, keeping his upper body in a stylish, fixed posture with controlled arm positions.
* **lower half body:** Hops on the avatar's right foot while the avatar's left leg kicks backward with a bent knee.
* **upper half body:** The avatar's right hand is held near the chin while the avatar's left arm extends backward.
* **torso:** Leans forward and orients slightly toward the avatar's right, remaining stable through the leg movements."""
    
    print("\n测试1: 完整回复（包含所有4个域）")
    is_valid, missing = check_response_domains(complete_response)
    print(f"  结果: {'通过✓' if is_valid else '失败✗'}")
    print(f"  缺失的域: {missing if missing else '无'}")
    
    # 测试2: 缺少一个域
    incomplete_response = """* **whole body:** The dancer leans forward and hops on his right foot.
* **lower half body:** Hops on the avatar's right foot.
* **upper half body:** The avatar's right hand is held near the chin."""
    
    print("\n测试2: 缺少 'torso' 域")
    is_valid, missing = check_response_domains(incomplete_response)
    print(f"  结果: {'通过✓' if is_valid else '失败✗'}")
    print(f"  缺失的域: {missing}")
    
    # 测试3: 自定义域列表
    custom_domains = ['whole body', 'lower half body']
    print("\n测试3: 自定义域列表 ['whole body', 'lower half body']")
    is_valid, missing = check_response_domains(incomplete_response, custom_domains)
    print(f"  结果: {'通过✓' if is_valid else '失败✗'}")
    print(f"  缺失的域: {missing if missing else '无'}")
    
    # 测试4: 空回复
    empty_response = "这是一段没有任何域标记的文本"
    print("\n测试4: 空回复（没有任何域）")
    is_valid, missing = check_response_domains(empty_response)
    print(f"  结果: {'通过✓' if is_valid else '失败✗'}")
    print(f"  缺失的域: {missing}")
    
    # 测试5: 大小写不敏感测试
    mixed_case_response = """* **WHOLE BODY:** Test
* **Lower Half Body:** Test
* **upper HALF body:** Test
* **TORSO:** Test"""
    
    print("\n测试5: 大小写混合")
    is_valid, missing = check_response_domains(mixed_case_response)
    print(f"  结果: {'通过✓' if is_valid else '失败✗'}")
    print(f"  缺失的域: {missing if missing else '无'}")
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)


def test_modifier_format():
    print("\n" + "=" * 80)
    print("Modifier 格式化功能测试")
    print("=" * 80)
    
    # 测试1: 奇数个modifiers（5个）- Pose比Trans多1个
    print("\n测试1: 5个modifiers（3个Pose + 2个Trans）")
    modifiers_5 = [
        "Initial standing position",
        "Transition to forward lean",
        "Forward lean with arm raise",
        "Transition to side step",
        "Final pose with arms extended"
    ]
    result = format_modifier_text(modifiers_5)
    print(result)
    print(f"\nPose数量: 3, Trans数量: 2")
    
    # 测试2: 偶数个modifiers（6个）
    print("\n" + "-" * 80)
    print("测试2: 6个modifiers（3个Pose + 3个Trans）")
    modifiers_6 = [
        "Initial standing position",
        "Transition to forward lean",
        "Forward lean with arm raise",
        "Transition to side step",
        "Side step position",
        "Transition to final pose"
    ]
    result = format_modifier_text(modifiers_6)
    print(result)
    print(f"\nPose数量: 3, Trans数量: 3")
    
    # 测试3: 只有1个modifier
    print("\n" + "-" * 80)
    print("测试3: 1个modifier（1个Pose）")
    modifiers_1 = ["Single pose description"]
    result = format_modifier_text(modifiers_1)
    print(result)
    print(f"\nPose数量: 1, Trans数量: 0")
    
    # 测试4: 空列表
    print("\n" + "-" * 80)
    print("测试4: 空列表")
    result = format_modifier_text([])
    print(f"结果: '{result}' (应该为空)")
    
    # 测试5: 大量modifiers（11个）
    print("\n" + "-" * 80)
    print("测试5: 11个modifiers（6个Pose + 5个Trans）")
    modifiers_11 = [f"Description {i+1}" for i in range(11)]
    result = format_modifier_text(modifiers_11)
    print(result)
    print(f"\nPose数量: 6, Trans数量: 5")
    
    print("\n" + "=" * 80)
    print("Modifier格式化测试完成！")
    print("=" * 80)


if __name__ == '__main__':
    test_domain_check()
    test_modifier_format()
