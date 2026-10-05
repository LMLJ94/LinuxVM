from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse

max_value = 255
max_value_H = 360//2
low_H = 0
low_S = 0
low_V = 0
high_H = max_value_H
high_S = max_value
high_V = max_value
window_capture_name = 'Original Image'
window_detection_name = 'HSV Filter Result'
trackbar_window_name = 'Trackbars'
window_histogram_name = 'Histogram (selected pixels)'
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'

def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H-1, low_H)
    cv.setTrackbarPos(low_H_name, trackbar_window_name, low_H)

def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H+1)
    cv.setTrackbarPos(high_H_name, trackbar_window_name, high_H)

def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S-1, low_S)
    cv.setTrackbarPos(low_S_name, trackbar_window_name, low_S)

def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S+1)
    cv.setTrackbarPos(high_S_name, trackbar_window_name, high_S)

def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V-1, low_V)
    cv.setTrackbarPos(low_V_name, trackbar_window_name, low_V)

def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V+1)
    cv.setTrackbarPos(high_V_name, trackbar_window_name, high_V)

def draw_values_panel(width=260, height=150):
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    rows = [
        (low_H_name, low_H), (high_H_name, high_H),
        (low_S_name, low_S), (high_S_name, high_S),
        (low_V_name, low_V), (high_V_name, high_V),
    ]
    for i, (name, value) in enumerate(rows):
        y = 25 + i * 22
        cv.putText(canvas, f'{name}: {value}', (10, y),
                   cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return canvas

def draw_hsv_histogram(hsv_image, mask, width=512, height=300):
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    channels = [(0, 180, (255, 100, 100), 'H'),
                (1, 256, (100, 255, 100), 'S'),
                (2, 256, (100, 100, 255), 'V')]

    for ch, bins, color, label in channels:
        hist = cv.calcHist([hsv_image], [ch], mask, [bins], [0, bins])
        cv.normalize(hist, hist, 0, height - 20, cv.NORM_MINMAX)
        bin_w = width / bins
        points = [(int(i * bin_w), height - int(hist[i])) for i in range(bins)]
        for i in range(1, len(points)):
            cv.line(canvas, points[i - 1], points[i], color, 1)

    for i, (_, _, color, label) in enumerate(channels):
        y = 20 + i * 20
        cv.line(canvas, (10, y), (30, y), color, 2)
        cv.putText(canvas, label, (35, y + 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return canvas

parser = argparse.ArgumentParser(description='HSV filtration for still images with trackbars.')
parser.add_argument('--image', help='Path to the input image.', default='/home/louise/Pictures/plant.jpg')
args = parser.parse_args()

frame = cv.imread(args.image)
if frame is None:
    print('Could not open or find the image:', args.image)
    exit(0)

frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

cv.namedWindow(window_capture_name)
cv.namedWindow(window_detection_name)
cv.namedWindow(trackbar_window_name)
cv.namedWindow(window_histogram_name)

cv.createTrackbar(low_H_name, trackbar_window_name, low_H, max_value_H, on_low_H_thresh_trackbar)
cv.createTrackbar(high_H_name, trackbar_window_name, high_H, max_value_H, on_high_H_thresh_trackbar)
cv.createTrackbar(low_S_name, trackbar_window_name, low_S, max_value, on_low_S_thresh_trackbar)
cv.createTrackbar(high_S_name, trackbar_window_name, high_S, max_value, on_high_S_thresh_trackbar)
cv.createTrackbar(low_V_name, trackbar_window_name, low_V, max_value, on_low_V_thresh_trackbar)
cv.createTrackbar(high_V_name, trackbar_window_name, high_V, max_value, on_high_V_thresh_trackbar)

cv.imshow(window_capture_name, frame)

while True:

    frame_threshold = cv.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
    frame_result = cv.bitwise_and(frame, frame, mask=frame_threshold)
    hist_image = draw_hsv_histogram(frame_HSV, frame_threshold)

    cv.imshow(window_detection_name, frame_result)
    cv.imshow(window_histogram_name, hist_image)
    cv.imshow(trackbar_window_name, draw_values_panel())

    key = cv.waitKey(30)
    if key == ord('q') or key == 27:
        break

cv.destroyAllWindows()
