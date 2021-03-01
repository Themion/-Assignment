# skin color detection
# created by SYO
# 2018.03.26

import cv2
import numpy as np
import math
import winsound as ws

img = cv2.imread("skin0.jpg")
converted = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

w_lower = np.array([0, 48, 170], dtype = "uint8")
w_upper = np.array([20, 255, 255], dtype = "uint8")
w_skinMask = cv2.inRange(converted, w_lower, w_upper)

b_lower = np.array([0, 48, 80], dtype = "uint8")
b_upper = np.array([20, 255, 180], dtype = "uint8")
b_skinMask = cv2.inRange(converted, b_lower, b_upper)

mix_skinMask = cv2.bitwise_and(w_skinMask, b_skinMask)

w_skin = cv2.bitwise_and(img, img, mask = w_skinMask)
b_skin = cv2.bitwise_and(img, img, mask = b_skinMask)
mix_skin = cv2.bitwise_or(w_skin, b_skin)

for i in range(mix_skinMask.shape[0]):
    for j in range(mix_skinMask.shape[1]):
        if mix_skinMask[i][j]:
            mix_skin[i][j][0] = 0
            mix_skin[i][j][1] = 0
            mix_skin[i][j][2] = 255
            
cv2.imshow("images", np.vstack([np.hstack([img, b_skin]), np.hstack([w_skin, mix_skin])]))
cv2.waitKey(0)
cv2.destroyAllWindows() 
