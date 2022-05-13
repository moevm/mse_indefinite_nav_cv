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

def run(frame):
    perspective = get_perspective(frame)

    left, center, right = crop_three_parts(perspective)

    perspective = match_by_template(right, perspective, (CENTER_END, 0), (255, 0, 0), RIGHT_MARKUP_TEMPLATE_PATH)
    perspective = match_by_template(center, perspective, (LEFT_END, 0), (0, 255, 0), CENTER_MARKUP_TEMPLATE_PATH)
    perspective = match_by_template(left, perspective, (0, 0), (0, 0, 255), LEFT_MARKUP_TEMPLATE_PATH)

    # cv2.imshow('left', left)
    # cv2.imshow('center', center)
    # cv2.imshow('right', right)
    cv2.imshow('perspective', perspective)
    cv2.imshow('original', frame)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
