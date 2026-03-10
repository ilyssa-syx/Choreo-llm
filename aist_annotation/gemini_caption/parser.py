#!/usr/bin/env python3
"""
解析 Gemini API 返回的 modifier 文本，将其从字符串格式转换为结构化的字典格式。

输入格式：
    "modifier": "* **whole body:** ...\n* **lower half body:** ...\n* **upper half body:** ...\n* **torso:** ..."

输出格式：
    "modifier": {
        "whole_body": "...",
        "lower_half_body": "...",
        "upper_half_body": "...",
        "torso": "..."
    }

如果文件中任何一个 motion 的 modifier 解析失败（缺少任何必需的 domain），
该文件将被跳过，不会生成输出。
"""

import json
import argparse
import re
from pathlib import Path


def parse_modifier_text(modifier_text, required_domains=None):
    """
    解析 modifier 文本，提取各个 domain 的描述
    
    Args:
        modifier_text: modifier 文本字符串
        required_domains: 必需的域列表（默认为 ['whole body', 'lower half body', 'upper half body', 'torso']）
    
    Returns:
        dict: 包含各 domain 描述的字典，如果解析失败返回 None
        缺失的 domain 列表（如果有）
    """
    if required_domains is None:
        required_domains = ['whole body', 'lower half body', 'upper half body', 'torso', 'simple tag']
    
    result = {}
    missing_domains = []
    
    # 先移除所有星号，简化匹配
    cleaned_text = modifier_text.replace('*', '')
    
    # 解析每个 domain
    for domain in required_domains:
        # 构建所有域名的正则匹配模式（用于向前查找下一个域名）
        all_domains_pattern = '|'.join([re.escape(d) for d in required_domains])
        
        # 匹配格式: domain名称 + 可选空白 + 换行 + 内容（直到下一个域名或结尾）
        # 忽略大小写匹配
        pattern = rf'{re.escape(domain)}\s*\n(.+?)(?=\n(?:{all_domains_pattern})|\Z)'
        match = re.search(pattern, cleaned_text, re.IGNORECASE | re.DOTALL)
        
        if match:
            description = match.group(1).strip()
            # 清理描述文本：移除多余的空格和换行符
            description = re.sub(r'\s+', ' ', description).strip()
            
            # 转换 domain 名称：空格变下划线
            key = domain.replace(' ', '_')
            result[key] = description
        else:
            # 缺失 domain
            missing_domains.append(domain)
    
    if missing_domains:
        return None, missing_domains
    
    return result, []


def parse_json_file(input_path, output_path, required_domains=None):
    """
    解析单个 JSON 文件
    
    Args:
        input_path: 输入 JSON 文件路径
        output_path: 输出 JSON 文件路径
        required_domains: 必需的域列表
    
    Returns:
        (success, error_message): 成功返回 (True, None)，失败返回 (False, error_message)
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 检查是否为列表
        if not isinstance(data, list):
            return False, "JSON 文件格式错误：根元素应该是数组"
        
        # 解析每个 motion 的 modifier
        for motion_data in data:
            motion_id = motion_data.get('motion', '?')
            modifier_text = motion_data.get('modifier', '')
            
            if not modifier_text:
                return False, f"motion {motion_id} 缺少 modifier 字段"
            
            # 如果 modifier 已经是字典格式，跳过解析
            if isinstance(modifier_text, dict):
                print(f"  注意: motion {motion_id} 的 modifier 已经是字典格式，跳过")
                continue
            
            # 解析 modifier
            parsed_modifier, missing_domains = parse_modifier_text(modifier_text, required_domains)
            
            if parsed_modifier is None:
                missing_str = ', '.join(missing_domains)
                return False, f"motion {motion_id} 的 modifier 解析失败，缺少以下 domain: {missing_str}"
            
            # 替换 modifier 字段
            motion_data['modifier'] = parsed_modifier
        
        # 保存到输出文件
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        
        return True, None
        
    except json.JSONDecodeError as e:
        return False, f"JSON 解析错误: {e}"
    except Exception as e:
        return False, f"处理文件时出错: {e}"


def main():
    parser = argparse.ArgumentParser(
        description='解析 Gemini 返回的 modifier 文本，将其转换为结构化字典格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python parser.py --input_folder ./raw_results --output_folder ./parsed_results
  python parser.py --input_folder ./raw_results --output_folder ./parsed_results --required_domains "whole body,lower half body,upper half body,torso"
  python parser.py --input_file ./raw_results/example.json --output_file ./parsed_results/example.json
        """
    )
    
    parser.add_argument('--input_folder', type=str, default=None,
                        help='输入 JSON 文件夹路径')
    parser.add_argument('--output_folder', type=str, default=None,
                        help='输出 JSON 文件夹路径')
    parser.add_argument('--input_file', type=str, default=None,
                        help='单个输入 JSON 文件路径（可选）')
    parser.add_argument('--output_file', type=str, default=None,
                        help='单个输出 JSON 文件路径（可选）')
    parser.add_argument('--required_domains', type=str, default=None,
                        help='必需的域列表，用逗号分隔（默认: "whole body,lower half body,upper half body,torso"）')
    
    args = parser.parse_args()
    
    # 验证参数
    if args.input_file and args.output_file:
        # 单文件模式
        single_file_mode = True
        input_path = Path(args.input_file)
        output_path = Path(args.output_file)
        
        if not input_path.exists():
            print(f"错误: 输入文件不存在: {input_path}")
            return
    elif args.input_folder and args.output_folder:
        # 批量处理模式
        single_file_mode = False
        input_folder = Path(args.input_folder)
        output_folder = Path(args.output_folder)
        
        if not input_folder.exists():
            print(f"错误: 输入文件夹不存在: {input_folder}")
            return
    else:
        print("错误: 请指定 --input_folder 和 --output_folder，或者 --input_file 和 --output_file")
        parser.print_help()
        return
    
    # 处理 required_domains
    required_domains = None
    if args.required_domains:
        required_domains = [d.strip() for d in args.required_domains.split(',')]
        print(f"使用自定义 domain 列表: {required_domains}")
    else:
        required_domains = ['simple tag', 'whole body', 'lower half body', 'upper half body', 'torso']
        print(f"使用默认 domain 列表: {required_domains}")
    
    print("=" * 80)
    print("Gemini Modifier 解析器")
    print("=" * 80)
    
    if single_file_mode:
        # 单文件处理
        print(f"\n处理文件: {input_path.name}")
        success, error_msg = parse_json_file(input_path, output_path, required_domains)
        
        if success:
            print(f"✓ 成功解析并保存到: {output_path}")
        else:
            print(f"✗ 解析失败: {error_msg}")
    else:
        # 批量处理
        json_files = list(input_folder.glob('*.json'))
        
        if not json_files:
            print(f"\n警告: 在 {input_folder} 中未找到 JSON 文件")
            return
        
        print(f"\n找到 {len(json_files)} 个 JSON 文件\n")
        
        success_count = 0
        failed_count = 0
        failed_files = []
        
        for json_file in sorted(json_files):
            output_path = output_folder / json_file.name
            print(f"处理: {json_file.name}")
            
            success, error_msg = parse_json_file(json_file, output_path, required_domains)
            
            if success:
                print(f"  ✓ 成功")
                success_count += 1
            else:
                print(f"  ✗ 失败: {error_msg}")
                print(f"  → 跳过输出")
                failed_count += 1
                failed_files.append((json_file.name, error_msg))
        
        print(f"\n{'='*80}")
        print(f"处理完成:")
        print(f"  成功: {success_count} 个")
        print(f"  失败: {failed_count} 个")
        
        if failed_files:
            print(f"\n失败的文件:")
            for filename, error_msg in failed_files:
                print(f"  - {filename}: {error_msg}")
        
        print(f"{'='*80}")


if __name__ == '__main__':
    main()
