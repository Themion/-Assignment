import numpy as np
import cv2

# Read the image file
image = cv2.imread('num (1).jpg')
image = cv2.resize(image, dsize=(360, 480), interpolation=cv2.INTER_AREA)

img = image.copy()

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.Canny(image, 170, 200)

cnts, _ = cv2.findContours(image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:30]

NumberPlateCnt = None

count = 0
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        NumberPlateCnt = approx
        break

stack = list()

for i in range(4):
    stack.append([NumberPlateCnt[i][0][0], NumberPlateCnt[i][0][1]])

for i in range(3):
    if stack[i][0] > stack[i + 1][0]:
         stack[i], stack[i + 1] = stack[i + 1], stack[i]

for i in range(3):
    if stack[i][0] > stack[i + 1][0]:
         stack[i], stack[i + 1] = stack[i + 1], stack[i]

if(stack[0][1] > stack[1][1]): stack[0], stack[1] = stack[1], stack[0]
if(stack[2][1] > stack[3][1]): stack[2], stack[3] = stack[3], stack[2]

stack = np.float32([[stack[0][0], stack[0][1]],
                    [stack[1][0], stack[1][1]],
                    [stack[2][0], stack[2][1]],
                    [stack[3][0], stack[3][1]]])

h = int(stack[1][1] - stack[0][1])
w = int(stack[2][0] - stack[0][0])

stack2 = np.float32([[0, 0], [0, h], [w, 0], [w, h]])

P = cv2.getPerspectiveTransform(stack, stack2)
plate = cv2.warpPerspective(img, P, (w, h))
cv2.imshow('number Plate', plate)
