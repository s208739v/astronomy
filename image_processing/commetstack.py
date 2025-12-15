import glob
import cv2
from matplotlib import pyplot as plt
import numpy as np
import scipy
from scipy import interpolate
import math
import time

def read_imgs(folder_path):
  files=glob.glob(folder_path)
  return files
"""
def crop(img,left_upper,h,w):
  
  x = int(left_upper[0])
  y = int(left_upper[1])
  print(x,y)
  return img[x:x+h, y:y+w, :]
"""

def crop(img, left_upper, h, w):
    # OpenCV準拠: left_upper = (x, y)
    x = int(left_upper[0])  # 列方向
    y = int(left_upper[1])  # 行方向
    print("Cropping (x, y) =", x, y)
    return img[y:y+h, x:x+w, :]

#ヒストグラムをプロットする
def plot_hist(img, channel, plot=None):
  img_hist_cv = cv2.calcHist([img], [channel], None, [256], [0, 256])
  if plot == True:    
    plt.plot(img_hist_cv)
    plt.show()
    print(len(img_hist_cv))
  
  
  pixel_num = img.shape[0]*img.shape[1]
  pixel_num_cumulative = 0
  min = None
  max = None
  for i in range(256):

    pixel_num_cumulative+=img_hist_cv[i]
    if pixel_num_cumulative/pixel_num >0.0001 and min == None:
      min = i
    if pixel_num_cumulative/pixel_num >0.9999 and max == None:
      max = i
      break
  print(min,max)
  #return img.min(), img.max()
  return min, max

#線形なトーンカーブ
def tone_curve_linear(min,max):
    a = 256/(max-min)
    b = -a*min
    tone_list = []
    for i in range(256):
        f = a*i+b
        if f < 0:
            tone_list.append(0)
        elif f > 255:
            tone_list.append(255)
        else:
            tone_list.append(f)
    return np.array(tone_list, dtype=np.uint8)

#背景推定に使う点を抽出
def extract_representative_points(img):
  x = []
  y= []
  z=[]
  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if i%100==0 and j%100==0:
          x.append(i)
          y.append(j)
          z.append(img[i,j,0])
  return x, y, z

def func_3(X, a,b, c,d,e,f,g, h, i ,j):
  x=X[0]
  y=X[1]
  return a*x**3+b*x**2+c*x + d*y**3+e*y**2+f*y+ g*x**2*y + h*x*y**2 + i*x*y +j

#背景推定と、減算をする
def estimate_and_subtrack_background(img, func):
  x,y,z = extract_representative_points(img)
  popt,pcov = scipy.optimize.curve_fit(func,np.array([y,x]),z)
  standa2 = [math.exp(-i**2/(2*50**2)) for i in range(256)]
  
  img2=img.copy()
  img3=img.copy()
  
  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
      for k in range(img.shape[2]):
        back_ground = func([j,i],popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8], popt[9])
        img3[i,j,k] = back_ground
        hosei = back_ground*standa2[int(abs(img[i,j,k]-back_ground))]
        #print(img.dtype,back_ground.dtype,hosei.dtype)
        if img[i,j,k] - hosei>0:
          img2[i,j,k] = img[i,j,k] - hosei
        else:
          img2[i,j,k] = 0
  return img2, img3

def low_freq(img):
  fft_img = np.fft.fft2(img)
  shift_fft_img=np.fft.fftshift(fft_img)
  
  h,w = shift_fft_img.shape
  R=10
  mask = np.zeros((h,w))
  for x in range(0,h):
    for y in range(0,w):
      if (x-h/2)**2 + (y-w)**2 >R**2:
        mask[x,y] = 1 

  img_fft_mask = shift_fft_img*mask
  shift_fft_img=np.fft.fftshift(img_fft_mask)
  img = np.fft.ifft2(shift_fft_img)
  img = img.real.astype(np.uint8)
  #cv2.imshow("fft", cv2.resize(img.astype(np.uint8), (int(img.shape[1]*0.5), int(img.shape[0]*0.5))))
  #cv2.waitKey(50)
  return np.stack((img, img,img),2)
  

def estimate_and_subtrack_background2(img, func):
  x,y,z = extract_representative_points(img)
  popt,pcov = scipy.optimize.curve_fit(func,np.array([y,x]),z)
  
  #for文回避
  base2x = np.tile(np.arange(img.shape[0],dtype=np.float64), (img.shape[1],1))
  base2y=np.tile(np.arange(img.shape[1],dtype=np.float64), (img.shape[0],1)).T
  base2 = popt[0]*base2x**3+popt[1]*base2x**2+popt[2]*base2x + popt[3]*base2y**3+popt[4]*base2y**2+popt[5]*base2y+ popt[6]*base2x**2*base2y + popt[7]*base2x*base2y**2 + popt[8]*base2x*base2y +popt[9]
  base2=base2.T
  base2 = np.stack([base2,base2,base2],axis=2)
  
  diff = np.abs(img-base2) #float64
  img = img - (base2*np.exp(-diff*diff/(2*200**2)))

  img[img < 0] = 0
  
  return img.astype(np.uint8), base2.astype(np.uint8)


def matching(img, templ, left_upper, area_size):
    imgs = [img]
    angles = [0]
    vals = []
    maxLocs = []
    theta_max = 0.01
    num = 10

    width = img.shape[1]
    height = img.shape[0]

    for i in range(int(num/2)):
        angle = (i+1)*theta_max/(num/2)

        M = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
        imgs.append(cv2.warpAffine(img.astype(np.float32), M, (width, height)))
        angles.append(angle)

        M = cv2.getRotationMatrix2D((width/2, height/2), -angle, 1)
        imgs.append(cv2.warpAffine(img.astype(np.float32), M, (width, height)))
        angles.append(-angle)

    for img_rot in imgs:
        result = cv2.matchTemplate(img_rot.astype(np.uint8), templ, cv2.TM_CCOEFF_NORMED)
        _, maxVal, _, maxLoc = cv2.minMaxLoc(result)
        vals.append(maxVal)
        maxLocs.append(maxLoc)

    max_index = np.argmax(vals)
    best_loc = maxLocs[max_index]
    best_angle = angles[max_index]

    # 🟢 OpenCV座標に合わせて (x, y) の順で返す
    return np.array([best_loc[0], best_loc[1]]), best_angle

def stack(img_base, img, dx, dy, angle, i):
    height, width = img.shape[:2]

    # 🟢 平行移動行列は [ [1, 0, dx], [0, 1, dy] ]
    M_translate = np.array([[1, 0, dx],
                            [0, 1, dy]], dtype=np.float32)

    M_rotate = cv2.getRotationMatrix2D((width/2, height/2), -angle, 1)

    # 🟢 順番を修正: 回転 → 平行移動
    img_warped = cv2.warpAffine(img.astype(np.float32), M_rotate, (width, height))
    img_warped = cv2.warpAffine(img_warped, M_translate, (width, height))

    img_base += img_warped
    preview = cv2.resize((img_base / (i + 1)).astype(np.uint8), (int(width * 0.5), int(height * 0.5)))
    cv2.imshow("stacked", preview)
    cv2.waitKey(50)

    return img_base

def select_roi_scaled(img, window_name="Select comet core", max_size=1000):
    """大きな画像でも画面に収まるようにスケールダウンしてROIを選択する"""
    h, w = img.shape[:2]
    scale = 1.0

    # もし画像が大きければ縮小
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img_resized = cv2.resize(img, (int(w * scale), int(h * scale)))
    else:
        img_resized = img.copy()

    # ROIを選択
    roi = cv2.selectROI(window_name, img_resized, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    # ROIが0の場合（キャンセル）
    if roi == (0, 0, 0, 0):
        print("ROI選択がキャンセルされました。")
        return None

    # 座標を元のスケールに戻す
    x, y, rw, rh = roi
    x = int(x / scale)
    y = int(y / scale)
    rw = int(rw / scale)
    rh = int(rh / scale)

    return (x, y, rw, rh)
  

files = read_imgs(r"D:\SharpCap Captures\2025-10-30\Capture\18_02_45\rawframes/*png*")
print(files)
img = cv2.imread(files[0])
img_base = np.zeros((img.shape[0],img.shape[1],3))


min_val,max_val = plot_hist(img,0)
tone_list_linear = tone_curve_linear(min_val, max_val)

img2 = cv2.LUT(img, tone_list_linear)
# ======== ここを追加 ========
# GUIでコマの部分を指定
print("最初のフレームから彗星のコマ部分を選択してください。")
roi = select_roi_scaled(img2, "Select comet core", max_size=1000)
if roi is None:
    exit()
x, y, w, h = roi

left_upper = np.array([x, y]) 
img_cropped = crop(img, left_upper, h, w)
cv2.imshow("Selected Template", img_cropped)
cv2.waitKey(500)
# ===========================


left_upper2 = left_upper.copy()  # ← 追加
for i in range(len(files)):
  #if i%10!=0:
  #  continue
  
  img = cv2.imread(files[i])
  print(files[i], img.shape)
  min,max = plot_hist(img,0)
  tone_list_linear = tone_curve_linear(min, max)

  img2 = cv2.LUT(img, tone_list_linear)
  
  img_subtracted = low_freq(img2[:,:,0])

  #img_subtracted, img_background2 = estimate_and_subtrack_background2(img2, func_3)
  
  #cv2.imshow("subtracted", cv2.resize(img_subtracted, (int(img_subtracted.shape[1]*0.5), int(img_subtracted.shape[0]*0.5))) )
  #cv2.waitKey(500)
  #cv2.imshow("", cv2.resize(img_background2, (int(img_subtracted.shape[1]*0.5), int(img_background2.shape[0]*0.5))) )
  #cv2.waitKey(2000)
  
  if i == 0:
    dx=0
    dy=0
    angle = 0
  else:
    left_upper2, angle = matching(img_subtracted, img_cropped, left_upper, np.array([600,600]))
    dx = left_upper2[0] - left_upper[0]
    dy = left_upper2[1] - left_upper[1]
    print("dx, dy, angle =", dx, dy, angle)
  
  img_base = stack(img_base, img_subtracted, -dx, -dy, angle, i)
  left_upper = left_upper2.copy()  # 🟢 更新を忘れずに！
  
  if i%20 == 0:
    cv2.imwrite(r"C:/Users/riopo/Downloads/"+str(i)+".png" ,(img_base/(i+1)).astype(np.uint8))

  img_cropped = crop((img_base/(i+1)).astype(np.uint8), left_upper, h, w)
  cv2.imshow("", img_cropped )
  cv2.waitKey(50)