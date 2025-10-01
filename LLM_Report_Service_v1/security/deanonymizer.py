# security/deanonymizer.py (最終完整版)

def deanonymize_report(report_text: str, reversal_map: dict) -> str:
    """
    使用還原映射表 (reversal_map)，將 LLM 生成的報告中的代碼
    (例如 "地點-A", "目標車輛A") 替換回真實名稱 (例如 "中壢區-B園區-內部", "BKE-6831")。

    Args:
        report_text: 從 LLM API 收到的、包含代碼的報告草稿字串。
        reversal_map: 一個字典，其鍵(key)為代碼，值(value)為真實名稱。
                      例如：{"地點-A": "楊梅區...", "目標車輛A": "BKE-6831"}

    Returns:
        一份完整的、人類可讀的最終報告字串。
    """
    deanonymized_text = report_text
    
    # 核心邏輯：為了避免替換順序出錯（例如，如果映射表中有 "地點-A" 和 "地點-AB"，
    # 直接替換可能會先將 "地點-AB" 中的 "地點-A" 換掉，導致錯誤），
    # 我們先將所有的代碼（keys）按照長度從長到短進行排序。
    # 這樣可以確保較長的、更具體的代碼被優先替換。
    
    sorted_keys = sorted(reversal_map.keys(), key=len, reverse=True)
    
    # 遍歷排序後的代碼列表，逐一進行替換
    for code in sorted_keys:
        real_name = reversal_map[code]
        deanonymized_text = deanonymized_text.replace(code, real_name)
        
    return deanonymized_text