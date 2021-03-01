import cv2
import time
import numpy as np

def emptyFunction(x):
    pass

def main():
    windowName = "Test"
    cv2.namedWindow(windowName)
    cap = cv2.VideoCapture(0)
    prevTime = 0

    if cap.isOpened():
        ret, ori = cap.read()
        
        mix = ori
        cv2.createTrackbar('Alpha', windowName, 333, 1000, emptyFunction)
        cv2.createTrackbar('MaskSize', windowName, 0, 10, emptyFunction)
        cv2.createTrackbar('Blending', windowName, 0, 3, emptyFunction)
    
        while ret:
            ret, ori = cap.read()
            can = cv2.Canny(ori, 100, 150, L2gradient=False)
            can = cv2.cvtColor(can, cv2.COLOR_GRAY2BGR)
            
            alpha = cv2.getTrackbarPos('Alpha', windowName) / 1000
            mix = cv2.addWeighted(ori, alpha, can, (1 - alpha), 0)

            #필터링 마스크 크기
            size = cv2.getTrackbarPos('MaskSize',windowName) * 2 + 1
            #필터링 타입
            tp = cv2.getTrackbarPos('Blending', windowName)

            if(tp == 1):    #일반 필터링
                kernel = np.ones((size, size), dtype = np.float32) / (size**2)
                mix = cv2.filter2D(mix, -1, kernel)
            elif(tp == 2):  #가우시안 필터링
                mix = cv2.GaussianBlur(mix, (size, size), 0)
            elif(tp == 3):  #블러 필터링
                mix = blur = cv2.blur(mix, (size, size))
###
            #현재 시간 가져오기 (초단위로 가져옴)
            curTime = time.time()
        
            #현재 시간에서 이전 시간을 빼면?
            #한번 돌아온 시간!!
            sec = curTime - prevTime
            #이전 시간을 현재시간으로 다시 저장시킴
            prevTime = curTime
        
            # 프레임 계산 한바퀴 돌아온 시간을 1초로 나누면 된다.
            # 1 / time per frame
            fps = 1/(sec)
        
            # 프레임 수를 문자열에 저장
            str = "FPS : %0.1f" % fps
        
            # 표시
            cv2.putText(ori, str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
###           
            cv2.imshow(windowName, np.hstack([ori, mix, can]))
            
            if cv2.waitKey(1) == 27:
                break

        cv2.destroyAllWindows()
        cap.release()
    
    else:
        ret = False

if __name__ == "__main__":
    main()
