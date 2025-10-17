# llm_clients/cloud_client.py (OpenAI 專用版 - 回傳 Token 用量)

import os
from openai import OpenAI
from dotenv import load_dotenv

# 從 prompts 模組匯入 SYSTEM_PROMPT (維持不變)
from prompts.report_prompt import SYSTEM_PROMPT

# --- 1. API 金鑰管理 (簡化版) ---
load_dotenv()
API_KEY = os.environ.get("OPENAI_API_KEY")

# --- 2. 初始化 OpenAI 客戶端 (簡化版) ---
try:
    if not API_KEY:
        raise ValueError("在 .env 檔案中找不到 OPENAI_API_KEY。")
    
    client = OpenAI(api_key=API_KEY)
    print("--- LLM Client: Initialized with OpenAI ---")

except Exception as e:
    print(f"初始化 OpenAI client 失敗: {e}")
    client = None

# --- 3. 修改函式以回傳完整的 response 物件 ---
def generate_report_from_summary(anonymized_summary_text: str):
    """
    將摘要發送給 OpenAI，並獲取包含 usage 的完整回覆。
    """
    if not client:
        raise ConnectionError("LLM API client 未成功初始化。")

    # 指定要使用的 OpenAI 模型
    model_to_use = "gpt-4o"

    try:
        print(f"--- Calling OpenAI model: {model_to_use} ---")
        response = client.chat.completions.create(
            model=model_to_use,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": anonymized_summary_text}
            ],
            temperature=0.2,
            max_tokens=2048
        )
        # 【【【 核心修改處：回傳完整的 response 物件 】】】
        return response.choices[0].message.content
    
    
    except Exception as e:
        print(f"呼叫 OpenAI API 時發生錯誤: {e}")
        return None # 發生錯誤時回傳 None