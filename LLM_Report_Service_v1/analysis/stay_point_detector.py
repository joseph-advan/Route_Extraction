# analysis/stay_point_detector.py (V2 更新版)

import pandas as pd
from datetime import timedelta

def find_stay_points_v2(vehicle_df_with_area: pd.DataFrame, time_threshold_minutes: int = 20) -> list:
    """
    從單一車輛的軌跡數據中，找出其停留點 (V2 版本)。
    V2 版本基於 'LocationAreaID' 而非 '攝影機名稱' 進行分群。

    Args:
        vehicle_df_with_area: 已合併了 'LocationAreaID' 的車輛軌跡 DataFrame。
        time_threshold_minutes: 定義「停留」所需的最短時間（分鐘）。

    Returns:
        一個包含停留點資訊的 list of dictionaries。
    """
    stay_points = []
    
    if vehicle_df_with_area.empty or 'LocationAreaID' not in vehicle_df_with_area.columns:
        print("錯誤：輸入的 DataFrame 缺少 'LocationAreaID' 欄位。")
        return stay_points

    # V2 核心改動：基於 'LocationAreaID' 進行分組
    grouped_by_area = vehicle_df_with_area.groupby(
        (vehicle_df_with_area['LocationAreaID'] != vehicle_df_with_area['LocationAreaID'].shift()).cumsum()
    )

    for _, group in grouped_by_area:
        if not group.empty:
            start_time = group['datetime'].iloc[0]
            end_time = group['datetime'].iloc[-1]
            duration = end_time - start_time
            
            duration_minutes = duration.total_seconds() / 60
            
            if duration_minutes >= time_threshold_minutes:
                # 為了讓報告更具可讀性，我們用這個區域裡拍到的第一支攝影機的名稱作為地點代表
                representative_location_name = group['攝影機名稱'].iloc[0]
                
                stay_points.append({
                    'location_area_id': group['LocationAreaID'].iloc[0],
                    'representative_name': representative_location_name,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration_minutes': round(duration_minutes, 2)
                })
                
    return stay_points