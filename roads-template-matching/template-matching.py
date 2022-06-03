import cv2
import numpy as np

PERSPECTIVE_RESUL_WIDTH = 260
PERSPECTIVE_RESUL_HEIGHT = 180

PERSPECTIVE_LEFT_TOP = (0, 162)
PERSPECTIVE_RIGHT_TOP = (640, 162)
PERSPECTIVE_RIGHT_BOTTOM = (640, 333)
PERSPECTIVE_LEFT_BOTTOM = (0, 333)

LEFT_END = 66
CENTER_END = 150
RIGHT_END = PERSPECTIVE_RESUL_WIDTH
PARTS_HEIGHT_LIMIT = 640

RIGHT_MARKUP_TEMPLATE_PATH = "right_template.png"
CENTER_MARKUP_TEMPLATE_PATH = "center_template.png"
LEFT_MARKUP_TEMPLATE_PATH = "left_template.png"
MATCH_METHOD = 0


def get_perspective(frame):
    pts1 = np.float32([PERSPECTIVE_LEFT_TOP, PERSPECTIVE_RIGHT_TOP, PERSPECTIVE_RIGHT_BOTTOM, PERSPECTIVE_LEFT_BOTTOM])
    pts2 = np.float32([[0, 0], [PERSPECTIVE_RESUL_WIDTH, 0], [PERSPECTIVE_RESUL_WIDTH, PERSPECTIVE_RESUL_HEIGHT], [0, PERSPECTIVE_RESUL_HEIGHT]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    perspective = cv2.warpPerspective(frame, matrix, (PERSPECTIVE_RESUL_WIDTH, PERSPECTIVE_RESUL_HEIGHT))
    return perspective

def crop_three_parts(frame):
    return frame[0:PARTS_HEIGHT_LIMIT, 0:LEFT_END], frame[0:PARTS_HEIGHT_LIMIT, LEFT_END:CENTER_END], frame[0:PARTS_HEIGHT_LIMIT, CENTER_END:RIGHT_END]

def match(frame, template_path):
    template = cv2.imread(template_path, cv2.IMREAD_COLOR)
    template = cv2.dilate(template, None)

    result = cv2.matchTemplate(frame, template, MATCH_METHOD)
    cv2.normalize(result, result, 0, 1, cv2.NORM_MINMAX, -1)

    _minVal, _maxVal, minLoc, maxLoc = cv2.minMaxLoc(result, None)
    if MATCH_METHOD == cv2.TM_SQDIFF or MATCH_METHOD == cv2.TM_SQDIFF_NORMED:
        matchLoc = minLoc
    else:
        matchLoc = maxLoc
    
    return matchLoc, template

def match_by_template(frame, frame_to_copy, offset, color, template_path):
    match_loc, template = match(frame, template_path)

    image_display = frame_to_copy.copy()
    match_loc = (match_loc[0] + offset[0], match_loc[1] + offset[1])
    cv2.rectangle(image_display, match_loc, (match_loc[0] + template.shape[1], match_loc[1] + template.shape[0]), color, 2, 8, 0)

    return image_display

def get_if_has_harris(frame, rectangle):
    left_top, right_bottom = rectangle[0], rectangle[1]

    for y in range(left_top[1], right_bottom[1]):
        for x in range(left_top[0], right_bottom[0]):
            if frame[y, x] == [0, 0, 255]:
                return rectangle
    
    return None


def get_template_rectangles(frame, a_template_path, b_template_path, c_template_path):
    a_match_loc, a_template = match(frame, a_template_path)
    b_match_loc, b_template = match(frame, b_template_path)
    c_match_loc, c_template = match(frame, c_template_path)

    a_rectangle = [a_match_loc, (a_match_loc[0] + a_template.shape[1], a_match_loc[1] + a_template.shape[0])]
    b_rectangle = [b_match_loc, (b_match_loc[0] + b_template.shape[1], b_match_loc[1] + b_template.shape[0])]
    c_rectangle = [c_match_loc, (c_match_loc[0] + c_template.shape[1], c_match_loc[1] + c_template.shape[0])]

    return get_if_has_harris(frame, a_rectangle), get_if_has_harris(frame, b_rectangle), get_if_has_harris(frame, c_rectangle)

def run(frame):
    gray = np.float32(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    harris = cv2.cornerHarris(gray, 2, 3, 0.04)
    frame[harris > 0.01 * harris.max()] = [0, 0, 255]

    perspective = get_perspective(frame)

    left, center, right = crop_three_parts(perspective)

    perspective = match_by_template(right, perspective, (CENTER_END, 0), (255, 0, 0), RIGHT_MARKUP_TEMPLATE_PATH)
    perspective = match_by_template(center, perspective, (LEFT_END, 0), (0, 255, 0), CENTER_MARKUP_TEMPLATE_PATH)
    perspective = match_by_template(left, perspective, (0, 0), (0, 0, 255), LEFT_MARKUP_TEMPLATE_PATH)

    all_template_rectangles = get_template_rectangles(frame, LEFT_MARKUP_TEMPLATE_PATH, CENTER_MARKUP_TEMPLATE_PATH, RIGHT_MARKUP_TEMPLATE_PATH)
    
    return perspective, frame, all_template_rectangles
