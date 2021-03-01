import numpy as np
import cv2

def emptyFunction(x):
    pass

def main():
    ori = cv2.imread("number.jpg", 0)
    img = ori.copy()

###
    lower = np.array([240], dtype = "uint8")
    upper = np.array([250], dtype = "uint8")
    img = cv2.inRange(img, lower, upper)
###
    img = cv2.GaussianBlur(img, (9, 9), 0)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    img = cv2.filter2D(img, -1, kernel)
###
    
    windowName = "Lines"
    cv2.namedWindow(windowName)
    cv2.createTrackbar('Thres1', windowName, 755, 2000, emptyFunction)
    cv2.createTrackbar('Thres2', windowName, 1000, 2000, emptyFunction)
    cv2.createTrackbar('THR', windowName, 100, 200, emptyFunction)

    while(True):
        thres1 = cv2.getTrackbarPos('Thres1', windowName)
        thres2 = cv2.getTrackbarPos('Thres2', windowName)
        thr = cv2.getTrackbarPos('THR', windowName)

        if(thres1 > thres2):
            thres1, thres2 = thres2, thres1

        can = cv2.Canny(img, thres1, thres2, None, 3)
        
        show = cv2.cvtColor(can, cv2.COLOR_GRAY2BGR)
        hough = cv2.HoughLinesP(can, 1, np.pi/180, thr, maxLineGap=10)
        
        if hough is not None:
            for line in hough:
                for x1, y1, x2, y2 in line:
                    cv2.line(show, (x1, y1), (x2, y2), [0, 0, 255], 3)
                    
        cv2.imshow(windowName, show)

        if cv2.waitKey(1) & 0xFF == 27:
            cv2.destroyAllWindows()
            break
    
if __name__ == "__main__":
    main()
