import cv2
import time
import os
import shutil
import datetime
import threading
import numpy as np
from pathlib import Path
from collections import deque
# --- 1. 画像処理クラス (RAW16対応版) ---
class Process_image():
    def __init__(self, min_length=50, FPS=10):
        self.min_length = min_length
        self.FPS = FPS
        
    def to_8bit(self, img):
        """16bit画像を8bitに変換するヘルパー関数"""
        if img.dtype == np.uint16:
            # 65535で割るのではなく、256で割ってビットシフトする（高速かつ一般的）
            return (img / 256).astype(np.uint8)
        return img

    def detect_line(self, gray_8bit_img):
        # 8bitのグレースケール画像を受け取って線を検出
        blur = cv2.GaussianBlur(gray_8bit_img, (5, 5), 0)
        canny = cv2.Canny(blur, 50, 150, 3) 
        lines = cv2.HoughLinesP(canny, 1, np.pi/180, threshold=20, 
                                minLineLength=self.min_length, maxLineGap=10)
        return lines

    def diff_and_merge(self, gray_img_list):
        """差分計算と明合成（リスト内の画像形式に合わせて処理）"""
        if len(gray_img_list) < 2:
            return None

        diff_list = []
        for i in range(len(gray_img_list) - 1):
            # 16bit同士の差分計算はOpenCVが対応しているのでそのままでOK
            diff = cv2.absdiff(gray_img_list[i], gray_img_list[i+1])
            diff_list.append(diff)
            
        if not diff_list:
            return None

        composite_img = diff_list[0]
        for i in range(1, len(diff_list)):
            composite_img = cv2.max(composite_img, diff_list[i])
            
        return composite_img

    def detect_meteor(self, img_list):
        """
        img_list: 16bitカラー(uint16) または 8bitカラー
        """
        if len(img_list) < 3:
            return False, None
        
        # 1. 解析用にグレースケール化（16bitのまま）
        gray_list = []
        for img in img_list:
            if len(img.shape) == 3: # カラーの場合
                gray_list.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            else:
                gray_list.append(img)

        # 2. 差分合成（16bitのまま計算することで階調飛びを防ぐ）
        composite_img_16 = self.diff_and_merge(gray_list)
        if composite_img_16 is None:
            return False, None
        
        # 3. 線分検出のために8bitへ変換
        # (Canny等のアルゴリズムは16bit入力を受け付けないため)
        composite_img_8 = self.to_8bit(composite_img_16)
        
        # 4. 検知実行
        detected = self.detect_line(composite_img_8)
        
        # 結果を返す（確認画像は8bit化したものを返す＝容量節約と視認性のため）
        if detected is not None:
            return True, composite_img_8
        else:
            return False, composite_img_8
            
    def save_movie(self, img_list, pathname):
        """動画保存（16bitが来たら8bitに変換して保存）"""
        if not img_list: return
        
        height, width = img_list[0].shape[:2]
        is_color = (len(img_list[0].shape) == 3)
        
        # 動画コンテナ(mp4)は通常8bitしか受け付けないため
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(str(pathname), fourcc, self.FPS, (width, height), isColor=is_color)
        
        for img in img_list:
            # 書き込み時に16bitなら8bitへ変換
            if img.dtype == np.uint16:
                frame = self.to_8bit(img)
            else:
                frame = img
            video.write(frame)
        video.release()

        
# --- 保存＆削除管理クラス (強化版) ---
class SaveManager:
    def __init__(self, base_save_path):
        self.base_save_path = Path(base_save_path)
        self.queue = deque()
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def add_task(self, img_list, file_paths, diff_img, delete_targets):
        with self.lock:
            # データのコピーを渡す
            self.queue.append((list(img_list), list(file_paths), diff_img.copy(), list(delete_targets)))

    def _worker(self):
        print("💾 保存マネージャー 待機中...")
        while self.running:
            task = None
            with self.lock:
                if self.queue:
                    task = self.queue.popleft()
            
            if task:
                try:
                    self._process_task(*task)
                except Exception as e:
                    print(f"❌ 保存タスク全体のエラー: {e}")
            else:
                time.sleep(0.5)

    def _process_task(self, img_list, file_paths, diff_img, delete_targets):
        # 【重要】ファイルロックが外れるのを待つため、処理開始前に少し待機
        # これだけで成功率が劇的に上がります
        time.sleep(2.0)

        now = datetime.datetime.now()
        save_dir = self.base_save_path / now.strftime('%Y-%m-%d') / now.strftime('%H-%M_%S')
        save_dir.mkdir(parents=True, exist_ok=True)
        raw_dir = save_dir / "raw"
        raw_dir.mkdir(exist_ok=True)
        
        print(f"📂 保存処理開始: {save_dir.name}")

        # 1. 検出画像と動画の保存 (これはメモリ上のデータなのでロック関係なし)
        try:
            cv2.imwrite(str(save_dir / "detection_composite.png"), diff_img)
            processor = Process_image()
            processor.save_movie(img_list, save_dir / "movie.mp4")
        except Exception as e:
            print(f"⚠️ 画像/動画書き出しエラー: {e}")

        # 2. 元画像のコピー (リトライ機能付き)
        for src_path in file_paths:
            self._copy_with_retry(src_path, raw_dir)

        # 3. 元ファイルの削除 (リトライ機能付き)
        # コピーが成功していようがいまいが、検知済みファイルは削除を試みる
        for del_path in delete_targets:
            self._remove_with_retry(del_path)
            
        print(f"✅ 保存完了: {save_dir.name}")

    def _copy_with_retry(self, src, dst_dir, max_retries=5):
        """しつこくコピーを試みる関数"""
        src_path = Path(src)
        if not src_path.exists():
            return # 既にないなら無視

        for i in range(max_retries):
            try:
                shutil.copy2(src, dst_dir)
                return # 成功したら終了
            except (PermissionError, OSError) as e:
                # ロックされていたら少し待って再挑戦
                time.sleep(0.5)
                if i == max_retries - 1:
                    print(f"❌ コピー失敗 (ロックされています): {src_path.name}")

    def _remove_with_retry(self, path, max_retries=5):
        """しつこく削除を試みる関数"""
        p = Path(path)
        if not p.exists():
            return

        for i in range(max_retries):
            try:
                os.remove(p)
                return # 成功
            except (PermissionError, OSError) as e:
                time.sleep(0.5)
                if i == max_retries - 1:
                    print(f"❌ 削除失敗 (ロックされています): {p.name}")

# --- 監視システムクラス ---
class FolderMonitorSystem:
    def __init__(self, watch_folder, save_folder, batch_size=30, overlap=5):
        self.watch_folder = Path(watch_folder)
        self.batch_size = batch_size
        self.overlap = overlap # 次のバッチに持ち越す枚数
        self.processor = Process_image(min_length=30)
        self.saver = SaveManager(save_folder)
        
        self.processed_files = set()
        # メモリ肥大化防止のため、処理済み履歴は一定数で古いものを捨てる
        self.processed_files_history = deque(maxlen=5000)

    def get_new_files(self):
        """
        フォルダ内の新しい画像を効率的に取得する
        os.scandir を使用して高速化
        """
        new_files = []
        try:
            # フォルダ内のエントリを走査
            with os.scandir(self.watch_folder) as entries:
                # pngのみ、かつ未処理のもの
                candidates = []
                for entry in entries:
                    if entry.is_file() and entry.name.lower().endswith('.jpg'):
                        if entry.path not in self.processed_files:
                            candidates.append(entry)
                
                # ファイル名でソート (SharpCap等はファイル名に時刻が入るためこれで順序保証可)
                # 最終更新日時(getmtime)より高速
                candidates.sort(key=lambda e: e.name)

                # 最新の1枚は書き込み中の可能性が高いため、安全のためスキップして次回に回す
                if len(candidates) > 1:
                    processing_candidates = candidates[:-1]
                else:
                    return []

                for entry in processing_candidates:
                    # サイズ0のファイルは無視（破損等の可能性）
                    if entry.stat().st_size > 0:
                        f_path = entry.path
                        new_files.append(f_path)
                        self.processed_files.add(f_path)
                        self.processed_files_history.append(f_path)
                
                # setの掃除
                if len(self.processed_files) > 5000:
                    # dequeから溢れた古い分をsetからも消したいが、
                    # 厳密な同期はコストが高いので、ここでは簡易的にhistoryに合わせてリフレッシュ等はしない
                    # 運用が数万枚を超えるなら set の定期クリアロジックを追加推奨
                    pass

        except Exception as e:
            print(f"⚠️ スキャンエラー: {e}")
        
        return new_files

    def load_image_old(self, path):
        """日本語パス対応の画像読み込み"""
        try:
            n = np.fromfile(path, np.uint8)
            img = cv2.imdecode(n, cv2.IMREAD_GRAYSCALE)
            return img
        except:
            return None
        
    def load_image(self, path):
            """RAW16 PNG対応の読み込み"""
            try:
                n = np.fromfile(path, np.uint8)
                
                # ★最重要変更点: 
                # IMREAD_UNCHANGED: ビット深度(16bit)もチャンネル数(Color)もそのまま読み込む
                # IMREAD_ANYDEPTH | IMREAD_ANYCOLOR: こちらの方が明示的で安全
                img = cv2.imdecode(n, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
                
                return img
            except:
                return None


    def run_monitor(self):
        print(f"👀 監視開始: {self.watch_folder}")
        print(f"   バッチサイズ: {self.batch_size}, オーバーラップ: {self.overlap}")

        img_buffer = []    # 画像データ(numpy)
        path_buffer = []   # ファイルパス(str)

        while True:
            new_paths = self.get_new_files()
            
            # 画像を読み込んでバッファに追加
            for path in new_paths:
                img = self.load_image(path)
                if img is not None:
                    img_buffer.append(img)
                    path_buffer.append(path)

            # バッファがバッチサイズに達したら解析実行
            if len(img_buffer) >= self.batch_size:
                
                # 解析対象の画像群
                current_imgs = img_buffer[:] # コピー
                current_paths = path_buffer[:]

                # 判定
                is_meteor, diff_img = self.processor.detect_meteor(current_imgs)

                # 削除対象の決定
                # オーバーラップ分（バッファの後ろの方）は、次の判定にも使うので削除してはいけない
                if self.overlap > 0:
                    files_to_delete = current_paths[:-self.overlap]
                else:
                    files_to_delete = current_paths[:]

                if is_meteor:
                    print(f"★ 流星検知！ ({datetime.datetime.now().strftime('%H:%M:%S')})")
                    # 検知時は保存マネージャへ投げる
                    # files_to_delete は「保存が終わった後に消していいファイル」として渡す
                    self.saver.add_task(
                        current_imgs, 
                        current_paths, 
                        diff_img,
                        files_to_delete
                    )
                else:
                    # 検知しなかった場合、不要なファイル（オーバーラップ以外）を即座に削除
                    for p in files_to_delete:
                        try:
                            os.remove(p)
                        except OSError:
                            pass # 既にない場合は無視
                
                # バッファの更新（オーバーラップ分だけ残してスライド）
                if self.overlap > 0:
                    img_buffer = img_buffer[-self.overlap:]
                    path_buffer = path_buffer[-self.overlap:]
                else:
                    img_buffer = []
                    path_buffer = []
                
                print(f"   処理完了: 残りバッファ {len(img_buffer)}枚")

            else:
                # 画像が足りない場合は少し待つ
                time.sleep(0.1)

if __name__ == "__main__":
    # --- 設定 ---
    # 監視元フォルダ (SharpCap等の保存先)
    TARGET_FOLDER = r"F:\SharpCap Captures\2025-12-14\test\19_40_40"
    
    # 検知データの保存先
    SAVE_FOLDER = r"F:\SharpCap Captures\2025-12-13\meteor"
    
    # 既存フォルダチェック
    if not os.path.exists(TARGET_FOLDER):
        print(f"❌ ターゲットフォルダが見つかりません: {TARGET_FOLDER}")
        # テスト用にフォルダを作る場合はコメントアウト解除
        # os.makedirs(TARGET_FOLDER, exist_ok=True)
    else:
        # batch_size: 一度に判定する枚数。30枚(約2秒分)程度推奨
        # overlap: 次の判定に持ち越す枚数。流星が切れ目に映った場合用。
        monitor = FolderMonitorSystem(
            TARGET_FOLDER, 
            SAVE_FOLDER, 
            batch_size=30, 
            overlap=1
        )
        
        try:
            monitor.run_monitor()
        except KeyboardInterrupt:
            print("\n🛑 終了します")