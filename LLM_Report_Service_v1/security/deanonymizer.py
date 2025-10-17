# security/deanonymizer.py (修正版 - 處理新的 reversal_map 結構)

def deanonymize_report(report_text: str, reversal_map: dict) -> str:
    """
    (修正版) 使用還原映射表 (reversal_map)，將 LLM 生成的報告中的代碼
    (例如 "Area-081", "目標車輛A") 替換回真實名稱。
    - 這個版本可以處理值為字典的 reversal_map。
    """
    deanonymized_text = report_text
    
    # 為了避免替換順序出錯（例如 "Area-1" 不會錯誤地替換到 "Area-10"），
    # 我們將所有的代碼（keys）按照長度從長到短進行排序。
    sorted_keys = sorted(reversal_map.keys(), key=len, reverse=True)
    
    # 遍歷排序後的代碼列表，逐一進行替換
    for code in sorted_keys:
        # 【【【 核心修正處 】】】
        # 從 reversal_map 中取得對應的資訊
        info = reversal_map[code]
        
        # 檢查 info 是否為字典，並取出 'name' 的值
        # 這樣 real_name 就永遠會是字串
        if isinstance(info, dict):
            real_name = info.get("name")
        else:
            # 為了相容舊格式，如果不是字典就直接使用
            real_name = info

        # 確保 real_name 不是 None 才進行替換
        if real_name is not None:
            deanonymized_text = deanonymized_text.replace(code, real_name)
        
    return deanonymized_text