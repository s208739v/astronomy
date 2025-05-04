import cv2

# rtspのURL指定でキャプチャするだけ
capture = cv2.VideoCapture('rtsp://6199:4003@192.168.137.20/live')

while(True):
    ret, frame = capture.read()
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()