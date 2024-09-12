import cv2 as cv
import numpy as np
import time

def filter_moving_points(good_old, good_new, threshold=1.0):
    distances = np.linalg.norm(good_new - good_old, axis=1)
    mask = distances > threshold
    return good_new[mask], good_old[mask]

def select_roi(frame):
    r = cv.selectROI("Select Robot", frame, fromCenter=False, showCrosshair=True)
    cv.destroyWindow("Select Robot")
    return r

cap = cv.VideoCapture('Robots.mp4')
# cap = cv.VideoCapture('Challenge.mp4')

orb = cv.ORB.create()
orb.setMaxFeatures(200)  # Reduced number of features

scale_factor = 0.5
fps_display_interval = 1
start_time = time.time()
frame_count = 0

image_container = None

ret, frame = cap.read()
frame = cv.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
robot_roi = select_roi(frame)

# Choose a potentially faster tracker if available
tracker = cv.TrackerKCF.create()
tracker.init(frame, robot_roi)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    frame_start_time = cv.getTickCount()
    frame = cv.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)

    success, robot_roi = tracker.update(frame)
    roi_image = frame[int(robot_roi[1]):int(robot_roi[1]+robot_roi[3]), int(robot_roi[0]):int(robot_roi[0]+robot_roi[2])]

    gray = cv.cvtColor(roi_image, cv.COLOR_BGR2GRAY)

    if image_container is not None:
        pts1 = cv.goodFeaturesToTrack(gray, maxCorners=200, qualityLevel=0.3, minDistance=5)  # Reduced feature count
        if pts1 is not None:
            pts1 = np.array(pts1).reshape(-1, 2)
            pts2, st, err = cv.calcOpticalFlowPyrLK(image_container, gray, pts1, None)
            st = st.ravel()
            pts2 = np.array(pts2).reshape(-1, 2)
            good_new = pts2[st == 1]
            good_old = pts1[st == 1]

            moving_new, moving_old = filter_moving_points(good_old, good_new, threshold=1.0)

            shifted_moving_new = moving_new + np.array([robot_roi[0], robot_roi[1]])
            shifted_moving_old = moving_old + np.array([robot_roi[0], robot_roi[1]])

            img_flow = frame.copy()
            for new, old in zip(shifted_moving_new, shifted_moving_old):
                a, b = new.ravel()
                c, d = old.ravel()
                img_flow = cv.line(img_flow, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
            
            # Display every few frames to reduce load
            if frame_count % 5 == 0:
                cv.imshow('Optical Flow Tracks - Selected Robot', img_flow)

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
