import numpy as np
import math
import cv2

col, width, row, height = -1, -1, -1, -1
frame = None
frame2 = None
inputmode = False
rectangle = False
trackWindow = None
sift = cv2.xfeatures2d.SIFT_create()

#찾을 타겟 이미지의 코너를 탐색
def getimg():
    global sift, kp1, des1, roi, h, w
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    h, w = roi.shape
    kp1, des1 = sift.detectAndCompute(roi, None)

def imgMatch(img2):
    global sift, kp1, des1, roi, h, w
    #비슷한 코너가 10개 이상 존재한다면 매칭에 성공한 것으로 간주
    MIN_MATCH_COUNT = 10

    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)     # img2: trainImage, 동영상의 프레임 하나

    #입력받은 프레임의 코너를 탐색하여 kd 트리를 이용해 코너를 매칭한다
    kp2, des2 = sift.detectAndCompute(img2, None)
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k = 2)
 
    # store all the good matches as per Lowe's ratio test.
    good = []
    dst = []

    #각 매칭점의 마할라노비스 거리를 비교해 좋은 매칭점을 골라낸다
    for m, n in matches:
        if m.distance < 1.0 * n.distance:
            good.append(m)

    #좋은 매칭점의 수가 MIN_MATCH_COUNT개보다 많거나 같다면
    if len(good) >= MIN_MATCH_COUNT:
        #좋은 매칭점을 한 컨테이너에 모은 뒤, 매칭점의 타입을 실행할 함수의 인자에 맞게 변환한다
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1, 1, 2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1, 1, 2)

        #RANSAC을 이용해 frame 속 타겟 이미지의 위치를 탐색한다
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        #타겟 이미지는 프레임에서 회전 혹은 변형 등의 왜곡이 있을 수 있으므로
        #perspectiveTransform을 통해 타겟 이미지를 프레임 내의 타겟 이미지와 일치하게 변형
        pts = np.float32([ [0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0] ]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        #타겟 이미지를 표시할 원의 중심을 저장
        dx = (dst[0][0][0] + dst[1][0][0] + dst[2][0][0] + dst[3][0][0]) / 4
        dy = (dst[0][0][1] + dst[1][0][1] + dst[2][0][1] + dst[3][0][1]) / 4

        #타겟 이미지의 위치와 원의 중심을 return한다
        return dst, dx, dy

    #좋은 매칭점의 수가 MIN_MATCH_COUNT개보다 적다면
    else:
        #좋은 매칭점의 수를 출력한다
        print ("Not enough matches are found: %d/%d" % (len(good), MIN_MATCH_COUNT))
        matchesMask = None

        #return값의 형식만 맞춰 쓸모없는 값을 return한다
        return None, 0, 0
        

#타겟 이미지를 만들 사각형을 만든다
def onMouse(event, x, y, flags, param):
    global col, width, row, height, frame, frame2, inputmode
    global rectangle, roi, trackWindow, xx, yy

    if inputmode:
        if event == cv2.EVENT_LBUTTONDOWN:
            rectangle = True
            col, row = x, y
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if rectangle:
                frame2 = frame.copy()
                cv2.rectangle(frame2, (col, row), (x, y), (0, 255, 0), 2)
                cv2.imshow('frame', frame2)

        elif event == cv2.EVENT_LBUTTONUP:
            inputmode = False
            rectangle = False
            frame2 = frame.copy()
            cv2.rectangle(frame2, (col, row), (x, y), (0, 255, 0), 2)
            height, width = abs(row - y), abs(col - x)
            trackWindow = (col, row, width, height)
            roi = frame[row : row + height, col : col + width]
            getimg()

    return

def camShift():
    global frame, frame2, inputmode, trackWindow, roi, out, width, height

    #속도 향상을 위해 타겟 이미지를 찾는 횟수를 제한한다
    frameSkip = 0

    #타겟 이미지를 표시할 원의 중심
    dx, dy = 0, 0
    #프레임 안에서의 타겟 이미지의 네 꼭지점
    dst = np.float32([ [0, 0], [0, 0], [0, 0], [0, 0] ]).reshape(-1, 1, 2)

    #dst, dx, dy값을 백업한다
    pdst, pdx, pdy = dst, dx, dy

    try:
        path ='video//1st_school_tour_studentcenter.avi'
        file = cv2.VideoCapture(path)
    except:
        print("Video Failed")
        return

    ret, frame = file.read()

    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame', onMouse, param = (frame, frame2))

    termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1)

    #동영상의 모든 프레임에 대해
    while True:
        ret, frame =  file.read()
        if not ret:
            break

        #처리 속도 향상을 위해 프레임의 사이즈를 조정한다
        frame = cv2.resize(frame, (640, 360))
        #원본 이미지와 출력할 이미지를 구분한다
        frame2 = frame.copy()

        #타겟 이미지가 선택되었다면
        if (trackWindow is not None):
            #현재 프레임이 이미지 매칭을 할 프레임이라면
            if (frameSkip == 0):
                #이미지 매칭을 실행
                dst, dx, dy = imgMatch(frame)

                #이미지 매칭에 성공하였다면 dst, dx, dy값을 백업
                if(dst is None):
                    dst = pdst
                    dx = pdx
                    dy = pdy
                #이미지 매칭에 실패하였다면 백업된 dst, dx, dy값을 복구
                else:
                    pdst = dst
                    pdx = dx
                    pdy = dy

                #다음 일정 프레임은 이미지 매칭을 실행하지 않는다
                frameSkip += 1

            #현재 프레임이 이미지 매칭을 할 프레임이 아니라면
            else:
                #frameSkip의 값을 1 올려준다
                frameSkip += 1
                #frameSkip이 3의 배수가 된다면 frameSkip의 값을 0으로 초기화해 이미지 매칭을 재실행하게끔 한다
                frameSkip = frameSkip % 3

            #이미지 매칭의 결과값을 frame2에 표시
            frame2 = cv2.polylines(frame2, [np.int32(dst)], True, (255, 255, 255), 2, cv2.LINE_AA)
            frame2 = cv2.circle(frame2, (math.floor(dx), math.floor(dy)), math.floor((width + height) / 4), (0, 255, 255), 2)

        #frame2를 출력
        cv2.imshow('frame', frame2)

        #이미지 매칭을 실행할 타겟 이미지는
        #동영상 재생 중에 i를 눌러 onMouse를 실행시킨 뒤
        #이미지 매칭을 할 프레임을 초기화
        k = cv2.waitKey(60) & 0xFF
        if k == 27:
            break

        if k == ord('i'):
            inputmode = True
            frameSkip = 0

            while inputmode:
                cv2.imshow('frame', frame)
                cv2.waitKey(0)

    file.release()
    cv2.destroyAllWindows()

camShift()
