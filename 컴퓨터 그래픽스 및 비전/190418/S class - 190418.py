import numpy as np
import cv2
        
def main():
    def mouse_callback(event, x, y, flags, param):
        nonlocal count
        nonlocal stack

        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(img, (x, y), 5, (0, 0, 0), 1)
            cv2.circle(img, (x, y), 4, (255, 255, 255), -1)
            stack[count][0] = x; stack[count][1] = y
            count = count + 1
    
    stack = np.float32([[0, 0], [0, 0], [0, 0], [0, 0]])
    count = 0
    
    img = cv2.imread("number2.jpg")
    ori = img.copy()
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', mouse_callback)
    
    lower = np.array([230, 230, 230], dtype = "uint8")
    upper = np.array([255, 255, 255], dtype = "uint8")
    Mask = cv2.inRange(ori, lower, upper)
    Mask = cv2.GaussianBlur(Mask, (5, 5), 0)
    cv2.imshow("find", Mask)

    while(count < 4):
        cv2.imshow('image',img)
        if cv2.waitKey(1) & 0xFF == 27:
            cv2.destroyAllWindows()
            break
    
    h = stack[1][1] - stack[0][1]
    w = stack[2][0] - stack[0][0]

    stack2 = np.float32([[0, 0], [0, h], [w, 0], [w, h]])

    print(stack)
    print(stack2)

    P = cv2.getPerspectiveTransform(stack, stack2)
    change = cv2.warpPerspective(ori, P, (w, h))
    cv2.imshow('change', change)
                
if __name__ == "__main__":
    main()
