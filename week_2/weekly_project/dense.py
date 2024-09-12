import cv2 as cv
import numpy as np
import time

def draw_flow(img, flow, step=16, scale=5):
    h, w = img.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    fx *= scale
    fy *= scale
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cv.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

image_container = None

# cap = cv.VideoCapture('Robots.mp4')
cap = cv.VideoCapture('Challenge.mp4')

scale_factor = 0.5
fps_display_interval = 1
start_time = time.time()
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    frame_start_time = cv.getTickCount()
    frame = cv.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    if image_container is not None:
        flow = cv.calcOpticalFlowFarneback(image_container, gray, None, 
                                           0.5, 3, 15, 3, 5, 1.2, 0)
        img_to_show = draw_flow(gray.copy(), flow, step=32, scale=5)
        cv.imshow('Optical Flow', img_to_show)

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
