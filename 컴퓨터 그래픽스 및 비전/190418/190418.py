# -*- coding: utf-8 -*-

import cv2
import numpy as np

imgpath = "number2.jpg"
original = cv2.imread(imgpath)

pts1 = np.float32([[0, 0], [0, 0], [0, 0], [0, 0]])
count = 0

def searching():
    global original

    converted = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 240], dtype = "uint8")
    upper = np.array([10, 10, 255], dtype = "uint8")
    
    Mask = cv2.inRange(converted, lower, upper)
    Mask = cv2.GaussianBlur(Mask, (5, 5), 0)
    
    cv2.imshow("find", Mask)

def change():
    global original, pts1

    h = pts1[1][1] - pts1[0][1]
    w = pts1[2][0] - pts1[0][0]

    pts2 = np.float32([[0, 0], [0, h], [w, 0], [w, h]])
    P = cv2.getPerspectiveTransform(pts1, pts2)

    changed = cv2.imread(imgpath)
    changed = cv2.warpPerspective(changed, P, (w, h))

    cv2.imshow("Changed", changed)

def CallBackFunc(event, x, y, flags, param):
    global original, pts1, count

    if event == cv2.EVENT_LBUTTONDOWN:
        pts1[count][0] = x
        pts1[count][1] = y

        cv2.circle(original, (x, y), 5, (0, 0, 0), 1)
        cv2.circle(original, (x, y), 4, (255, 255, 255), -1)

        count += 1

def CallBackVoid(event, x, y, flags, param):
    pass

def main():
    global original, pts1, count
    
    windowName = "Original"
    cv2.namedWindow(windowName)
    cv2.setMouseCallback(windowName, CallBackFunc)

    searching()

    while(True):
        cv2.imshow(windowName, original)

        if count >= 4:
            cv2.setMouseCallback(windowName, CallBackVoid)  #에러 방지
            change()

        k = cv2.waitKey(1) & 0xFF
        if k == 27: break

    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()
