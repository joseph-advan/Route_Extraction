# analysis/anomaly_detector.py (Final Version - Enhanced Anomaly Details)

import pandas as pd
import numpy as np

def find_anomalies_v3(trips_df: pd.DataFrame, regular_patterns: list) -> dict:
    """
    (V3) Detects path and duration anomalies from all trips.
    This version returns the full details for each individual anomaly event.
    """
    anomalies = {
        "infrequent_patterns": [],
        "duration_anomalies": []
    }

    if trips_df.empty:
        return anomalies

    # --- 1. Path Anomaly Detection ---
    regular_signatures = {p['signature'] for p in regular_patterns}
    infrequent_trips_df = trips_df[~trips_df['signature'].isin(regular_signatures)]

    if not infrequent_trips_df.empty:
        for _, row in infrequent_trips_df.iterrows():
            anomalies["infrequent_patterns"].append({
                "start_time": row['start_time'],
                "end_time": row['end_time'], # <--- ADDED
                "start_area_id": row['start_area_id'],
                "end_area_id": row['end_area_id'],
                "duration_minutes": row['duration_minutes'],
                "signature": row['signature']
            })

    # --- 2. Duration Anomaly Detection ---
    pattern_groups = {p['signature']: trips_df[trips_df['signature'] == p['signature']] for p in regular_patterns}
    
    for pattern in regular_patterns:
        signature = pattern['signature']
        group = pattern_groups[signature]
        durations = group['duration_minutes']
        
        if len(durations) < 4:
            continue

        median_duration = durations.median()
        Q1 = durations.quantile(0.25)
        Q3 = durations.quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = group[durations > upper_bound]
        
        for _, outlier_trip in outliers.iterrows():
            anomalies["duration_anomalies"].append({
                "start_time": outlier_trip['start_time'],
                "end_time": outlier_trip['end_time'], # <--- ADDED
                "pattern_signature": signature,
                "actual_duration_minutes": round(outlier_trip['duration_minutes'], 2),
                "median_duration_for_pattern": round(median_duration, 2)
            })

    return anomalies