import cv2
import os
import numpy as np
from pathlib import Path


def create_meteor_shower_movie(source_folder, output_filename):
    """
    指定フォルダ以下のすべての movie.mp4 を探し出し、
    長さが違っても対応可能な比較明合成動画を作成する。
    """
    source_path = Path(source_folder)
    # サブフォルダも含めて movie.mp4 を全て探す
    video_files = list(source_path.rglob("movie.mp4"))
    
    if not video_files:
        print(f"❌ 動画が見つかりません: {source_folder}")
        return

    print(f"🎬 {len(video_files)} 本の動画を検出。合成を開始します...")

    # 合成用のフレームバッファ（ここに重ねていく）
    master_frames = [] 
    
    # 最初の動画からFPS（再生速度）とサイズを取得する
    first_cap = cv2.VideoCapture(str(video_files[0]))
    fps = first_cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 15.0 # 取得失敗時のデフォルト
    
    width = int(first_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(first_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first_cap.release()

    print(f"   設定: {width}x{height}, FPS={fps}")
    
    count = 0
    for video_path in video_files:
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            continue
            
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break # この動画が終了したらループを抜ける（短い動画はここで終わる）
            
            # 安全策: 解像度が違う動画が混ざっていたらリサイズする
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))

            # --- 長さ違い対応ロジック ---
            if frame_idx >= len(master_frames):
                # マスターバッファより長い部分（または最初の動画）
                # 単純にリストの後ろに追加していく
                master_frames.append(frame.copy())
            else:
                # 既にデータがある時間帯
                # 現在の合成結果と、新しい動画のフレームを比較して明るい方を取る
                master_frames[frame_idx] = cv2.max(master_frames[frame_idx], frame)
            
            frame_idx += 1
            
        cap.release()
        count += 1
        print(f"   合成完了 ({count}/{len(video_files)}): ...{str(video_path)[-30:]}")

    if not master_frames:
        print("❌ 合成できるフレームがありませんでした。")
        return

    # 結果の書き出し
    print(f"💾 合計 {len(master_frames)} フレームの動画を書き出し中...")
    
    # カラーか白黒か判定
    is_color = (len(master_frames[0].shape) == 3)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height), isColor=is_color)
    
    for frame in master_frames:
        out.write(frame)
        
    out.release()
    print(f"✅ 完成しました！保存先: {output_filename}")

if __name__ == "__main__":
    # --- 設定 ---
    # 流星動画が保存されている親フォルダ（日付フォルダの親）を指定してください
    # 例: r"F:\SharpCap Captures\Meteor_Detections"
    TARGET_FOLDER = r"F:\SharpCap Captures\2025-12-13\meteor\2025-12-14" 
    
    # 出力ファイル名
    OUTPUT_FILE = "All_Meteors_Flowing.mp4"
    
    if os.path.exists(TARGET_FOLDER):
        create_meteor_shower_movie(TARGET_FOLDER, OUTPUT_FILE)
    else:
        print(f"❌ フォルダが見つかりません: {TARGET_FOLDER}")