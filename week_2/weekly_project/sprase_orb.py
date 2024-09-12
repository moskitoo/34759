import cv2 as cv
import numpy as np
import time


def filter_moving_points(good_old, good_new, threshold=1.0):
    distances = np.linalg.norm(good_new - good_old, axis=1)
    return good_new[distances > threshold], good_old[distances > threshold]

# cap = cv.VideoCapture('Robots.mp4')
cap = cv.VideoCapture('Challenge.mp4')

# sift = cv.SIFT_create()
orb = cv.ORB.create()
orb.setMaxFeatures(300)

scale_factor = 0.5

fps_display_interval = 1
start_time = time.time()
frame_count = 0

image_container = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    frame_start_time = cv.getTickCount()
    frame = cv.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    if image_container is not None:
        # kp1, des1 = sift.detectAndCompute(image_container, None)
        kp1 = orb.detect(gray,None)
        kp1, orb_des_1 = orb.compute(gray, kp1)
        if kp1:
            pts1 = np.array([kp.pt for kp in kp1], dtype=np.float32)
            pts2, st, err = cv.calcOpticalFlowPyrLK(image_container, gray, pts1, None)
            st = st.ravel()
            good_new = pts2[st == 1]
            good_old = pts1[st == 1]
            moving_new, moving_old = filter_moving_points(good_old, good_new, threshold=1.0)
            img_flow = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
            for new, old in zip(moving_new, moving_old):
                a, b = new.ravel()
                c, d = old.ravel()
                img_flow = cv.line(img_flow, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
            cv.imshow('Optical Flow Tracks', img_flow)

    image_container = gray.copy()
    frame_end_time = cv.getTickCount()
    elapsed_time = (frame_end_time - frame_start_time) / cv.getTickFrequency()
    frame_count += 1
    current_time = time.time()
    if current_time - start_time >= fps_display_interval:
        fps = frame_count / (current_time - start_time)
        print(f'FPS: {fps:.2f}')
        start_time = current_time
        frame_count = 0

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
