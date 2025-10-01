# analysis/trip_segmenter.py (V3 - 基於時間間隔的新邏輯)

import pandas as pd

def segment_trips_v3(vehicle_df: pd.DataFrame, gap_threshold_minutes: int = 20) -> list:
    """
    (V3) 根據軌跡點之間的時間間隔，將車輛的軌跡切割成一段段的「行程」。
    這個版本不再依賴於預先計算好的「長時停留點」，因此更加穩健。

    Args:
        vehicle_df: 預處理過的、單一車輛的 DataFrame (已按時間排序)。
        gap_threshold_minutes: 定義一次移動結束所需的時間間隔（分鐘）。

    Returns:
        一個包含行程資訊的 list of dictionaries。
    """
    trips = []
    
    if vehicle_df.empty:
        return trips

    # --- V3 核心邏輯 ---
    # 1. 計算每一筆連續紀錄之間的時間差
    time_gaps = vehicle_df['datetime'].diff()

    # 2. 找出所有時間差超過閾值的點，這些點是「行程的斷點」
    #    .shift(-1) 是為了將斷點標記在上一筆紀錄，代表「此處為終點」
    trip_breakpoints = time_gaps > pd.Timedelta(minutes=gap_threshold_minutes)
    
    # 3. 使用 .cumsum() 技巧，為每一次連續的移動（即一次行程）分配一個唯一的 ID
    trip_ids = trip_breakpoints.cumsum()
    
    # 4. 根據行程 ID 進行分組
    grouped_by_trip = vehicle_df.groupby(trip_ids)

    for trip_id, group in grouped_by_trip:
        # 一個有效的行程至少需要 2 個點（起點和終點）
        if len(group) > 1:
            start_point = group.iloc[0]
            end_point = group.iloc[-1]
            
            # 從原始的攝影機紀錄中提取行程資訊
            trip_start_time = start_point['datetime']
            trip_end_time = end_point['datetime']
            duration = trip_end_time - trip_start_time
            
            # 從 group 中提取起點和終點的區域 ID
            start_area_id = start_point['LocationAreaID'] if 'LocationAreaID' in start_point else 'Unknown'
            end_area_id = end_point['LocationAreaID'] if 'LocationAreaID' in end_point else 'Unknown'

            trips.append({
                'start_time': trip_start_time,
                'end_time': trip_end_time,
                'duration_minutes': round(duration.total_seconds() / 60, 2),
                'start_area_id': start_area_id,
                'end_area_id': end_area_id,
                'start_location_name': start_point['攝影機名稱'],
                'end_location_name': end_point['攝影機名稱'],
                'point_count': len(group),
                'path_camera_names': group['攝影機名稱'].tolist()
            })
            
    return trips