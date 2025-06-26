import glob
import cv2
from matplotlib import pyplot as plt
import numpy as np
import scipy
from scipy import interpolate
import math
import time
import os
from threading import Lock
lock = Lock()
import shutil
import threading
import datetime
import socket
import sys
import atexit

ip = "127.0.0.1"
port = 8000

class Process_image():
    
    def __init__(self):
        self.min_length = 30
        self.save_path = ""
        
    def detect_line(self, img, min_length):
        """画像上の線状のパターンを流星として検出する。
        Args:
        img: 検出対象となる画像
        min_length: HoughLinesPで検出する最短長(ピクセル)
        Returns:
        検出結果
        """
        blur_size = (5, 5)
        blur = cv2.GaussianBlur(img, blur_size, 0)
        canny = cv2.Canny(blur, 100, 200, 3)

        # The Hough-transform algo:
        return cv2.HoughLinesP(canny, 1, np.pi/180, 25, minLineLength=min_length, maxLineGap=5)


    def diff(self, img_list, mask=None):
        """画像リストから差分画像のリストを作成する。

        Args:
        img_list: 画像データのリスト
        mask: マスク画像(2値画像)

        Returns:
        差分画像のリスト
        """
        diff_list = []
        if mask is None:
            mask = np.zeros(img_list[0].shape, dtype=np.uint8)

        for img1, img2 in zip(img_list[:-2], img_list[1:]):
            if mask is not None:
                img1 = cv2.bitwise_or(img1, mask)
                img2 = cv2.bitwise_or(img2, mask)
            diff_list.append(cv2.subtract(img1, img2))

        return diff_list

    def brightest(self, img_list):
        """比較明合成処理
        Args:
        img_list: 画像データのリスト

        Returns:
        比較明合成された画像
        """
        output = img_list[0]

        for img in img_list[1:]:
            output = cv2.max(img, output)

        return output


    def detect_meteor(self, img_list):
        """img_listで与えられた画像のリストから流星(移動天体)を検出する。
        """

        if len(img_list) > 2:
            # 差分間で比較明合成を取るために最低3フレームが必要。
            # 画像のコンポジット(単純スタック)
            diff_img = self.brightest(self.diff(img_list, None))
            detected = self.detect_line(diff_img, self.min_length)
            if detected is not None:
                '''
                for meteor_candidate in detected:
                    print('{} {} A possible meteor was detected.'.format(obs_time, meteor_candidate))
                '''
                print("A possible meteor was detected.")
                #cv2.imwrite(path_name, self.composite_img)

                """
                # 検出した動画を保存する。
                movie_file = str(
                    Path(self.save_path, "movie-" + filename + ".mp4"))
                self.save_movie(img_list, movie_file)
                """
                return True, diff_img
            else:
                return False, diff_img
                
    def save_movie(self, img_list, pathname):
        #画像リストから動画を作成する。

        size = (self.WIDTH, self.HEIGHT)
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

        video = cv2.VideoWriter(pathname, fourcc, self.FPS, size)
        for img in img_list:
            video.write(img)
        video.release()
            
class cameras():
    def __init__(self, path, num):
        self.image_processer = Process_image()
        
        #--- Sharpcap使用時に、画像保存に使用する ---
        self.files_to_be_del = []#削除する（流星が移っていないファイル）
        self.files_to_be_saved = []#保存するファイル（写ってるやつ）
        
        #--- atomcam rtsp時に、画像保存に使用する ---
        self.frames_to_be_saved = [] #保存する画像データ（検出画像or定時記録）
        
        
        self.folders_to_be_saved = [] #各写真を保存するフォルダ名を入れる（時刻）        
        self.folder_path=path #保存フォルダ
        self.num=num #１回の検知処理に使うフレーム数
        
        self.regular_record_interval = 60*60*0 + 60*10 + 0 #定時記録の間隔を秒で指定
        
        #定時記録を保存するパス
        self.regular_record_path = self.folder_path + "/regular_record"
        try:
            os.mkdir(self.regular_record_path)
        except:
            pass
        
        self.start_time = "22:00" #撮影開始時刻
        self.end_time = "04:00" #撮影終了時刻
        self.current_is_night = False
        self.rtsp_url="rtsp://6199:4003@192.168.137.144/live"
    
    #夜間のみ動くように判定を入れる   
    def is_night(self): 
        """夜間の場合にTrueを返す"""
        now = datetime.datetime.now().time()
        return now >= datetime.datetime.strptime(self.start_time, "%H:%M").time() or now < datetime.datetime.strptime(self.end_time, "%H:%M").time()
    
    #フレームを保存する(opencvでダイレクトにカメラ画像にアクセスするとき)
    def organize_frames(self):
        while(True):
            time.sleep(0.2)
            if len(self.frames_to_be_saved)!=0:
                self.save_frames()
    
    #フレームを保存する(opencvでダイレクトにカメラ画像にアクセスするとき)
    def save_frames(self):
        with lock:
            ftbs = self.frames_to_be_saved.copy()
            fotbs = self.folders_to_be_saved.copy()
            fitbs = self.files_to_be_saved.copy()
            self.frames_to_be_saved = []
            self.folders_to_be_saved = []
            self.files_to_be_saved = []
            
        for i in range(len(ftbs)): 
            try:   
                os.mkdir(fotbs[i])
            except:
                pass
            for j in range(len(ftbs[i])):
                try:
                    cv2.imwrite(fotbs[i]+"/"+str(fitbs[i][j])+".png" ,ftbs[i][j])
                    #cv2.imwrite(fotbs[i]+"/"+str(fitbs[i][j])+".jpg" ,ftbs[i][j], [cv2.IMWRITE_JPEG_QUALITY, 80])
                except:
                    pass
    
    
    #画像ファイルの読み込み(sharpcap)     
    def read_imgs(self, folder_path):
        files=glob.glob(folder_path)
        return files
    
    
    #ファイルの保存と削除（sharpcap）
    def organize_files(self):
        while True:
            time.sleep(0.1)
            #print(self.files_to_be_del)
            if len(self.files_to_be_del)!=0:
                self.delete_pictures()
            if len(self.files_to_be_saved)!=0:
                self.save_detected_result()

    
    #ファイル削除の関数(sharpcap)
    def delete_pictures(self):
        with lock:
            ftpd = self.files_to_be_del.copy()
            self.files_to_be_del = []
        for path in ftpd:
            os.remove(path)
    
    #検知画像の保存（sharpcap）
    def save_detected_result(self):
        with lock:
            ftbs = self.files_to_be_saved.copy()
            fotbs = self.folders_to_be_saved.copy()
            self.files_to_be_saved = []
            self.folders_to_be_saved = []
        for i in range(len(ftbs)): 
            try:   
                os.mkdir(fotbs[i])
            except:
                pass
            for file in ftbs[i]:
                try:
                    shutil.move(file, fotbs[i])
                except:
                    pass
            
            
            
    #撮影済みデータから見つけるとき
    def folder_camera(self, folder_path, num):
        
        files = read_imgs(folder_path + "/*png*")
        img = cv2.imread(files[0])
        img_base = np.zeros((img.shape[0],img.shape[1],3))

        img_list=[]
        for i in range(len(files)):
            img = cv2.imread(files[i])
            img_list.append(img)
            #所定枚数毎に検知をする
            
            if i!=0 and i%num==0:
                self.image_processer.detect_meteor()
                img_list = []
    
    #ファイル名からindex読み取り(sharpcap)
    def read_file_index(self, filename):
        file_index = int((filename.split("_")[-1]).replace(".png", ""))
        return file_index

    def read_file_base(self, filename):
        file_base = filename.split("_")[0]
        return file_base
    
    #RTSP対応カメラ用
    def rtsp_live(self):
        capture = cv2.VideoCapture(self.rtsp_url)
        #sharpcapがフォルダに保存した画像をリアルタイムに取得して処理する関数
        folder_path=self.folder_path
        
        num=self.num
        img_list=[]
        last_regular_record_time = datetime.datetime.now() #最後に経過を記録した時間
        
        while(1):
            try:
                if self.is_night() == False and self.current_is_night == False: #昼間
                    continue
                elif self.is_night() == False and self.current_is_night == True: #夜から朝になったとき、接続を切る
                    self.current_is_night = False
                    print("Night is over!!" )
                    capture.release()
                    continue
                elif self.is_night() == True and self.current_is_night == False: #昼間から夜に変わったとき、再接続。古いフレームが残っていないように数フレーム分無視
                    self.current_is_night = True
                    capture.release()
                    print("Night has come!!" )
                    time.sleep(1)
                    capture = cv2.VideoCapture(self.rtsp_url)
                    #古いイメージを捨てる
                    for _ in range(100):
                        ret, _ = capture.read()
                        if not ret:
                            break
                    continue
                                
                dt_now = datetime.datetime.now() #現在時刻取得
                ret, frame = capture.read()

                # ret が False の場合、または frame が None の場合、フレームの読み込みに失敗               
                if not ret or frame is None:
                    print(f"フレームの読み込みに失敗しました。接続が切れた可能性があります。5秒後に再接続を試みます。 Time: {datetime.datetime.now()}")
                    capture.release()  # 現在のキャプチャオブジェクトを解放
                    time.sleep(5)      # カメラやネットワークが復旧するのを待つ
                    capture = cv2.VideoCapture(self.rtsp_url) # 再接続を試みる
                    print("再接続を試行しました。")
                    img_list.clear() # 溜まっていたリストをクリア
                    continue # ループの先頭に戻る      

                img_list.append(frame)
                #検知に必要な枚数がたまったら処理開始
                if len(img_list)>num-1:

                    result, diff_img = self.image_processer.detect_meteor(img_list)       
                    
                    cv2.imshow("", diff_img )
                    cv2.waitKey(1)
                    
                    if result == True:#検知した時
                        with lock:
                            self.frames_to_be_saved.append(img_list)
                            self.folders_to_be_saved.append(self.folder_path + "/"+ dt_now.strftime('%d-%H_%M_%S'))
                            self.files_to_be_saved.append(np.arange(0,num))
                    img_list = []

                #定時記録
                if (dt_now - last_regular_record_time).seconds > self.regular_record_interval:
                    with lock:
                        self.frames_to_be_saved.append([frame])
                        self.folders_to_be_saved.append(self.regular_record_path)
                        self.files_to_be_saved.append([dt_now.strftime('%d-%H_%M_%S')])
                    last_regular_record_time = dt_now
                time.sleep(0.01)   
    
            except Exception as e:
                print(f"Error in while loop. Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Error detail: {e}")
                
                print("img_list=",img_list)
                for i in range(len(img_list)):
                    print("img=",img_list[i])
                
                capture.release()
                time.sleep(120)
                capture = cv2.VideoCapture(self.rtsp_url)

    
    #sharpcapがフォルダに保存した画像をリアルタイムに取得して処理する関数
    def sharpcap_live(self):
        
        folder_path=self.folder_path
        
        num=self.num
        img_list=[]
        last_processed_picture_index = self.read_file_index(self.read_imgs(folder_path + "/*png*")[0]) #最後に検知処理された画像のindex
        print(last_processed_picture_index)
        last_regular_record_time = datetime.datetime.now() #最後に経過を記録した時間
        regular_recorded_file=""#削除予定ファイル名に追加しないようにするために必要
        
        while(1):
            try:
                if self.is_night() == False:
                    continue
            except:
                pass
            try:
                files = self.read_imgs(folder_path + "/*png*")
                
                latest_picture_index = self.read_file_index(files[-1])
                file_base = self.read_file_base(files[-1]) #写真ファイル名のインデックスの前の部分

                img_list = []
                read_files = [] #読み込むファイル

                dt_now = datetime.datetime.now() #現在時刻取得



                #検知に必要な枚数がたまったら処理開始
                if (latest_picture_index - last_processed_picture_index) >num+1:
                    #print(latest_picture_index, last_processed_picture_index, latest_picture_index - last_processed_picture_index )
                    for i in range(num):
                        filename = folder_path+"/tes_"+str(last_processed_picture_index+i)+".png"
                        #print(filename)
                        read_files.append(filename)
                        img = cv2.imread(filename)
                        im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        img_list.append(im_gray)

                    result, diff_img = self.image_processer.detect_meteor(img_list)
                    last_processed_picture_index +=num             
                    
                    cv2.imshow("", diff_img )
                    cv2.waitKey(1)
                    
                    img_list.clear()
                    #print(latest_picture_index, last_processed_picture_index, latest_picture_index - last_processed_picture_index )
                    
                    if result == True:#検知した時
                        with lock:
                            self.files_to_be_saved.append(read_files)
                            self.folders_to_be_saved.append(self.folder_path + "/"+ dt_now.strftime('%d-%H_%M_%S'))
                    #定時記録
                    if (dt_now - last_regular_record_time).seconds > self.regular_record_interval and len(files)!=0:
                        with lock:
                            self.files_to_be_saved.append([files[-1]])
                            self.folders_to_be_saved.append(self.regular_record_path)
                        regular_recorded_file=files[-1]
                        last_regular_record_time = dt_now 
                                        
                    if result == False:#検知されなかったとき
                        if regular_recorded_file in read_files:
                            read_files.remove(regular_recorded_file) #定時記録用写真は削除しないように除く
                        with lock:
                            self.files_to_be_del+=read_files #削除予定のファイルに追加
            except:
                print("error in loop")
            
            time.sleep(0.01)

    #botプロセスと通信する関数。スレッド立てて動かす
    def socket_client(self):
        print("Connected!!!!!")

        while True:
            try:
                print("<メッセージを入力してください>")
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect((ip, port))
                message = input('>>>')
                if not message:
                    s.send("quit".encode("utf-8"))
                    break
                s.send(message.encode("utf-8"))
                s.close()
            except Exception as e:
                print(e)
                time.sleep(1)

    #全体のプロセスを実行する関数
    def process(self, camera="a"):
        if camera=="s": #sharpcap
            thread1 = threading.Thread(target=self.sharpcap_live)
            thread2 = threading.Thread(target=self.organize_files)
        elif camera=="a": #atomcam
            thread1 = threading.Thread(target=self.rtsp_live)
            thread2 = threading.Thread(target=self.organize_frames)     
        thread3 = threading.Thread(target=self.socket_client)
        
        thread1.start()
        thread2.start()
        #thread3.start()
        
        thread1.join()
        thread2.join()
        #thread3.join()

def communicate_bot():
    #botのプロセスと通信する
    pass

#ChatGPTが考えたログを出力するやつ
class Tee:
    def __init__(self, log_file_path):
        self.terminal = sys.__stdout__
        self.log = open(log_file_path, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

# ログファイル作成
log_filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".log"
sys.stdout = sys.stderr = Tee(log_filename)

atexit.register(sys.stdout.close)

# 入力プロンプト表示
print("folder path=", end="", file=sys.__stdout__, flush=True)
path = input()
print("num for detection", end="", file=sys.__stdout__, flush=True)
num = int(input())
camera = cameras(path, num)
camera.process("a")
