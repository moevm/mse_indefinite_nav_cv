import cv2
import sys

VIDEO_PATH = "video.mp4"
CORNER_TEMPLATE_PATH = "corner.jpeg"
ROAD_TEMPLATE_PATH = "road.jpeg"
MATCH_METHOD = 0

def match(frame, template_path):
    template = cv2.imread(template_path, cv2.IMREAD_COLOR)
    result = cv2.matchTemplate(frame, template, MATCH_METHOD)
    cv2.normalize(result, result, 0, 1, cv2.NORM_MINMAX, -1)

    _minVal, _maxVal, minLoc, maxLoc = cv2.minMaxLoc(result, None)
    if MATCH_METHOD == cv2.TM_SQDIFF or MATCH_METHOD == cv2.TM_SQDIFF_NORMED:
        matchLoc = minLoc
    else:
        matchLoc = maxLoc
    
    return matchLoc, template

def match_by_template(frame):
    corner_match_loc, corner_template = match(frame, CORNER_TEMPLATE_PATH)
    road_match_loc, road_template = match(frame, ROAD_TEMPLATE_PATH)

    image_display = frame.copy()

    cv2.rectangle(image_display, corner_match_loc, (corner_match_loc[0] + corner_template.shape[1], corner_match_loc[1] + corner_template.shape[0]), (0, 255, 0), 2, 8, 0)
    cv2.rectangle(image_display, road_match_loc, (road_match_loc[0] + road_template.shape[1], road_match_loc[1] + road_template.shape[0]), (0, 0, 255), 2, 8, 0)

    return image_display

def get_harris(frame):
    frame_copy = frame.copy()
    gray = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
    dst = cv2.cornerHarris(gray, 3, 3, 0.04)
    dst = cv2.dilate(dst, None)
    frame_copy[dst > 0.2 * dst.max()] = [0, 0, 255]
    return frame_copy

def main():
    video = cv2.VideoCapture(VIDEO_PATH)

    while True:
        ret, frame = video.read()

        harris = get_harris(frame)
        matched_frame = match_by_template(frame)
        cv2.imshow('harris', harris)
        cv2.imshow('template matching', matched_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
