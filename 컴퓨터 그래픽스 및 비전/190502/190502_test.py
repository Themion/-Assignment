import numpy as np
import cv2

def emptyFunction(x):
    pass

def main():
    ori = cv2.imread("number.jpg", 0)

    windowName = "Lines"
    cv2.namedWindow(windowName)

    cv2.createTrackbar('Thres1', windowName, 755, 2000, emptyFunction)
    cv2.createTrackbar('Thres2', windowName, 1000, 2000, emptyFunction)
    cv2.createTrackbar('THR', windowName, 100, 200, emptyFunction)
    cv2.createTrackbar('Length', windowName, 50, 100, emptyFunction)

    while(True):
        thres1 = cv2.getTrackbarPos('Thres1', windowName)
        thres2 = cv2.getTrackbarPos('Thres2', windowName)
        thr = cv2.getTrackbarPos('THR', windowName)
        lth = cv2.getTrackbarPos('Length', windowName)

        if(thres1 > thres2):
            thres1, thres2 = thres2, thres1

        #흰색 번호판 추출 및 중간값 블러링
        #salt & pepper 오류 제거
        img = cv2.inRange(ori, 240, 255)
        img = cv2.medianBlur(img, 5)

        #Contour를 이용해 번호판 안쪽 공간을 모두 메운다
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for i in range(0, len(contours)):
            cv2.drawContours(img, contours, i, 255, -1)

        #열기 기법으로 번호판이 아닌 큰 덩어리를 제거
        kernel = np.ones((11, 11), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

        #캐니 에지 - 허프 라인으로 번호판의 에지 탐색
        #필요하다면 HpoughLineP를 HoughLine으로 바꿈
        img = cv2.Canny(img, thres1, thres2, None, 3)
        hough = cv2.HoughLinesP(img, 1, np.pi/180, thr, minLineLength = lth, maxLineGap=10)
        
        if hough is not None:
            for line in hough:
                for x1, y1, x2, y2 in line:
                    cv2.line(img, (x1, y1), (x2, y2), 127, 3)

        #제대로 된 에지를 찾았을 경우 각 선분 간의 교점으로 네 점을 찾아야 함

        cv2.imshow(windowName, img)
        
        if cv2.waitKey(1) & 0xFF == 27:
            cv2.destroyAllWindows()
            break
    
if __name__ == "__main__":
    main()
