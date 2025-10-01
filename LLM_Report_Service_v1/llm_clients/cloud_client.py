# llm_clients/cloud_client.py (重構版)

import os
from openai import OpenAI
from dotenv import load_dotenv

# 從我們的新檔案中，匯入 SYSTEM_PROMPT 變數
from prompts.report_prompt import SYSTEM_PROMPT

# --- 1. API 金鑰管理 ---
load_dotenv()
API_KEY = os.environ.get("OPENAI_API_KEY")

# 初始化 OpenAI 客戶端
try:
    if not API_KEY:
        raise ValueError("OPENAI_API_KEY not found. Please set it in your .env file or environment variables.")
    client = OpenAI(api_key=API_KEY)
except Exception as e:
    print(f"初始化 OpenAI client 失敗: {e}")
    client = None

# --- 2. 系統提示 (System Prompt) ---
# 現在這裡變得非常乾淨，直接從外部匯入即可。

def generate_report_from_summary(anonymized_summary_text: str) -> str:
    """
    將去識別化的分析摘要發送給雲端 LLM，並獲取報告。
    """
    if not client:
        error_message = "錯誤：OpenAI API client 未成功初始化。請檢查您的 .env 檔案或環境變數中的 API 金鑰設定。"
        print(error_message)
        return f"{error_message}\n\n--- 模擬報告 ---\n摘要內容顯示...\n{anonymized_summary_text[:300]}..."

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT}, # <-- 直接使用匯入的變數
                {"role": "user", "content": anonymized_summary_text}
            ],
            temperature=0.2,
            max_tokens=2048
        )
        return response.choices[0].message.content

    except Exception as e:
        error_message = f"呼叫 OpenAI API 時發生錯誤: {e}"
        print(error_message)
        return f"生成報告時發生錯誤：{e}"