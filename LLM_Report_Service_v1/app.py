# app.py (V12 - 完整穩定版)

import pandas as pd
from pathlib import Path
import sys

# ==========================================
# 匯入各個分析模組
# ==========================================
# 1. 單一車輛 LLM 報告服務
from reporting_service import run_llm_reporting_flow

# 2. 隨行車輛/跟隨分析 (保留原有功能)
from analysis.convoy_analyzer import run_trip_oriented_convoy_analysis

# 3. 雙車碰面分析 (新功能)
# 請確保 analysis/meeting_analyzer.py 檔案存在且已更新
from analysis.meeting_analyzer import run_dual_vehicle_meeting_analysis

def main_console():
    """
    應用主控台：負責資料載入與主選單邏輯
    """
    # ==============================================================================
    # 步驟 1: 載入並預處理資料
    # ==============================================================================
    full_data = None
    try:
        current_file_path = Path(__file__)
        project_root_path = current_file_path.parent
        
        # 設定資料路徑 (請確認檔名是否正確)
        DATA_FILE_PATH = project_root_path / 'data' / 'realistic_vehicle_dataset1.csv'
        
        print(f"正在讀取資料: {DATA_FILE_PATH} ...")
        
        if not DATA_FILE_PATH.exists():
            print(f"錯誤：找不到檔案 {DATA_FILE_PATH}")
            print("請確認檔案是否已放入 data 資料夾中。")
            input("按 Enter 鍵離開...")
            return

        full_data = pd.read_csv(DATA_FILE_PATH)
        
        # --- 資料清洗與格式化 ---
        # 1. 時間格式轉換
        full_data['datetime'] = pd.to_datetime(full_data['日期'] + ' ' + full_data['時間'])
        full_data = full_data.sort_values(by='datetime').reset_index(drop=True)
        
        # 2. 確保 LocationID 為字串
        if 'LocationID' in full_data.columns:
            full_data['LocationID'] = full_data['LocationID'].astype(str)
        else:
            print("警告：資料中缺少 'LocationID' 欄位，可能會影響部分分析功能。")
        
        # 3. 【關鍵修正】強制轉換經緯度為浮點數
        # 這是為了確保後續計算距離 (Haversine) 時不會因為資料含有字串而失敗
        if '經度' in full_data.columns and '緯度' in full_data.columns:
            full_data['經度'] = pd.to_numeric(full_data['經度'], errors='coerce')
            full_data['緯度'] = pd.to_numeric(full_data['緯度'], errors='coerce')
            
            # 移除座標無效 (NaN) 的資料，避免髒資料導致程式崩潰
            before_len = len(full_data)
            full_data = full_data.dropna(subset=['經度', '緯度'])
            after_len = len(full_data)
            if before_len != after_len:
                print(f"已移除 {before_len - after_len} 筆經緯度無效的資料。")
            
        print("--- 成功讀取並預處理軌跡資料 ---")
        print(f"有效資料筆數: {len(full_data)}")

    except Exception as e:
        print(f"\n[嚴重錯誤] 讀取資料時發生例外狀況: {e}")
        input("按 Enter 鍵離開...")
        return

    # ==============================================================================
    # 步驟 2: 顯示功能菜單並接收使用者選擇
    # ==============================================================================
    while True:
        try:
            print("\n" + "="*50)
            print("== 車輛軌跡智慧分析系統 ==")
            print("="*50)
            print("  [1] 單一車輛軌跡分析 (LLM 報告)")
            print("  [2] 分析目標行程的隨行車輛 (行程導向)")
            print("  [3] 雙車碰面分析 (Dual-Vehicle Meeting)")
            print("  [q] 結束程式")
            
            choice = input("請輸入您的選擇: ").strip()
            
            # --- 選項 1: 單一車輛分析 ---
            if choice == '1':
                run_single_vehicle_analysis(full_data)
            
            # --- 選項 2: 隨行車輛分析 (保留原功能) ---
            elif choice == '2':
                run_trip_oriented_convoy_analysis(full_data)
                
            # --- 選項 3: 雙車碰面分析 (新功能) ---
            elif choice == '3':
                run_dual_vehicle_analysis_flow(full_data)
                
            # --- 離開 ---
            elif choice.lower() == 'q':
                print("感謝使用，程式結束。")
                break
            else:
                print("無效的選擇，請重新輸入。")
        except KeyboardInterrupt:
            print("\n程式已強制中斷。")
            break
        except Exception as e:
            print(f"\n[執行錯誤] 發生未預期的錯誤: {e}")
            print("請檢查您的資料或程式碼設定。")
            # 不中斷迴圈，讓使用者可以重試別的功能

def run_single_vehicle_analysis(full_data):
    """處理「單一車輛報告生成」的使用者互動與呼叫"""
    available_plates = sorted(full_data['車牌'].unique())
    print("\n--- 生成單一車輛深度分析報告 ---")
    
    # 分頁顯示車牌，避免洗版
    display_limit = 20
    print(f"資料集中共有 {len(available_plates)} 輛車。前 {display_limit} 筆：")
    for i, plate in enumerate(available_plates[:display_limit]): 
        print(f"  [{i+1}] {plate}")
    if len(available_plates) > display_limit:
        print(f"  ... (還有 {len(available_plates) - display_limit} 台車)")
    
    try:
        choice_input = input(f"請選擇要分析的車牌 [1-{len(available_plates)}]: ").strip()
        if not choice_input: return # 若直接按 Enter 則返回

        choice = int(choice_input)
        if 1 <= choice <= len(available_plates):
            target_plate = available_plates[choice - 1]
            debug_choice = input("是否啟用除錯模式 (y/N)? ").lower()
            debug_mode = True if debug_choice == 'y' else False
            
            # 呼叫報告服務
            run_llm_reporting_flow(full_data, target_plate, debug_mode=debug_mode)
        else:
            print("錯誤：輸入的編號超出範圍。")
            
    except ValueError:
        print("錯誤：請輸入有效的數字編號。")
    except IndexError:
        print("錯誤：索引錯誤。")

def run_dual_vehicle_analysis_flow(full_data):
    """
    處理「雙車碰面分析」的使用者互動 (優化版 - 支援列表選擇)
    """
    print("\n--- 雙車碰面分析模式 ---")
    
    available_plates = sorted(full_data['車牌'].unique())
    total_plates = len(available_plates)
    
    # 顯示車輛列表
    DISPLAY_LIMIT = 30
    print(f"資料庫中共有 {total_plates} 輛車。請依序選擇兩輛車進行比對。")
    print("-" * 30)
    for i, plate in enumerate(available_plates):
        if i < DISPLAY_LIMIT:
            print(f"  [{i+1:02d}] {plate}")
        else:
            break   
    if total_plates > DISPLAY_LIMIT:
        print(f"  ... (還有 {total_plates - DISPLAY_LIMIT} 輛車)")
    print("-" * 30)

    # 內部函式：處理使用者選擇邏輯
    def get_plate_choice(prompt_msg, exclude_plate=None):
        while True:
            user_input = input(prompt_msg).strip()
            if not user_input: return None # 允許取消
            
            # 1. 嘗試解析為數字編號
            if user_input.isdigit():
                idx = int(user_input) - 1
                if 0 <= idx < total_plates:
                    chosen = available_plates[idx]
                    if chosen == exclude_plate:
                        print(f"錯誤：車牌 {chosen} 已經被選擇過了，請選擇另一輛。")
                        continue
                    return chosen
                else:
                    print(f"錯誤：編號超出範圍 (1-{total_plates})。")
            
            # 2. 嘗試直接輸入車牌字串
            else:
                if user_input in available_plates:
                    if user_input == exclude_plate:
                        print(f"錯誤：車牌 {user_input} 已經被選擇過了，請選擇另一輛。")
                        continue
                    return user_input
                else:
                    print(f"錯誤：找不到車牌 '{user_input}'。")

    # --- 獲取第一台車 ---
    plate_a = get_plate_choice("請選擇第一台車 (輸入編號或車牌): ")
    if not plate_a: return
    print(f"-> 已選擇 Car A: {plate_a}")
    
    # --- 獲取第二台車 ---
    plate_b = get_plate_choice("請選擇第二台車 (輸入編號或車牌): ", exclude_plate=plate_a)
    if not plate_b: return
    print(f"-> 已選擇 Car B: {plate_b}")

    print(f"\n即將開始分析：【{plate_a}】 vs 【{plate_b}】...")
    
    # 篩選出這兩台車的資料
    df_a = full_data[full_data['車牌'] == plate_a].copy()
    df_b = full_data[full_data['車牌'] == plate_b].copy()

    # 呼叫後端分析邏輯
    # (注意：這裡的 run_dual_vehicle_meeting_analysis 會自動處理 LocationAreaID)
    run_dual_vehicle_meeting_analysis(df_a, df_b, plate_a, plate_b)

if __name__ == '__main__':
    main_console()