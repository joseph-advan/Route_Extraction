# analysis/camera_clusterer.py (修正 FutureWarning)

import pandas as pd
import numpy as np

def haversine_distance(lon1, lat1, lon2, lat2):
    """
    計算兩個經緯度座標點之間的距離（單位：公尺）。
    """
    R = 6371000  # 地球半徑，單位為公尺
    
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    
    a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    meters = R * c
    return meters

def cluster_cameras_by_distance(all_cameras_df: pd.DataFrame, radius_meters: int = 50) -> pd.DataFrame:
    """
    根據地理距離對所有攝影機進行分群。
    """
    cameras_with_clusters = all_cameras_df.copy()
    # ========================= 程式碼修正處 =========================
    # 初始化時，直接將欄位類型設定為 object (可以存放文字)，並使用 None 作為未分群的標記
    cameras_with_clusters['LocationAreaID'] = pd.Series(dtype='object')
    # =========================  修正結束  =========================
    
    cluster_id_counter = 0
    
    for index, camera in cameras_with_clusters.iterrows():
        # 如果這支攝影機已經被分過群，就跳過
        if pd.notna(camera['LocationAreaID']):
            continue
            
        current_cluster_id = f"Area-{cluster_id_counter:03d}"
        cameras_with_clusters.loc[index, 'LocationAreaID'] = current_cluster_id
        
        # 找出所有其他尚未分群的攝影機
        unclustered_cameras = cameras_with_clusters[pd.isna(cameras_with_clusters['LocationAreaID'])]
        
        if not unclustered_cameras.empty:
            distances = haversine_distance(
                camera['經度'], camera['緯度'],
                unclustered_cameras['經度'], unclustered_cameras['緯度']
            )
            nearby_indices = unclustered_cameras.index[distances <= radius_meters]
            cameras_with_clusters.loc[nearby_indices, 'LocationAreaID'] = current_cluster_id
        
        cluster_id_counter += 1
        
    return cameras_with_clusters