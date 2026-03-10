import http.client
import json

API_KEY = "sk-rXnFhVdisEL92jDqiCJelD36fonFQH6ogmoVKGeA0CRP3ubf"

def call_gemini_api(system, prompt, model):
    conn = http.client.HTTPSConnection("jeniya.top")
    payload = json.dumps({
        "system_instruction": {
            "parts": [{"text": system}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}]
            }
        ]
    })
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }
    conn.request("POST", f"/v1beta/models/{model}:generateContent", payload, headers)
    res = conn.getresponse()
    data = res.read()
    response_json_str = data.decode("utf-8")
        
    # 解析响应
    try:
        response_dict = json.loads(response_json_str)
        
        # 检查是否有错误
        if "error" in response_dict:
            print(f"API 错误: {response_dict['error']}")
            return None
        
        candidates = response_dict.get("candidates", [])
        if not candidates:
            print("没有找到候选结果")
            return None
            
        first_candidate = candidates[0]
        content_parts = first_candidate.get("content", {}).get("parts", [])
            
        # 拼接所有parts的文本
        model_text = "".join([part.get("text", "") for part in content_parts])
        
        if not model_text.strip():
            print("返回内容为空")
            return None
            
        return model_text
            
    except Exception as e:
        print(f"解析模型回复出错: {e}\nRaw response: {response_json_str}")
        return None  # 直接返回 None，不要返回错误信息字符


