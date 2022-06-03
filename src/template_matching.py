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
    return perspective, matrix

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

def match_by_template(frame, offset, color, template_path):
    match_loc, template = match(frame, template_path)

    match_loc = np.array([match_loc[0] + offset[0], match_loc[1] + offset[1]])
    rect = np.array([match_loc, (match_loc[0] + template.shape[1], match_loc[1] + template.shape[0])])
    return [(rect[0] + (rect[1] - rect[0]) / 2)]

def run(frame):
    perspective, matrix = get_perspective(frame)

    left, center, right = crop_three_parts(perspective)

    l = match_by_template(left, (0, 0), (0, 0, 255), LEFT_MARKUP_TEMPLATE_PATH)
    c = match_by_template(center, (LEFT_END, 0), (0, 255, 0), CENTER_MARKUP_TEMPLATE_PATH)
    r = match_by_template(right, (CENTER_END, 0), (255, 0, 0), RIGHT_MARKUP_TEMPLATE_PATH)
    inv_trans = np.linalg.pinv(matrix)
    l[0] = cv2.perspectiveTransform(np.array([[[l[0][0], l[0][1]]]]), inv_trans)
    c[0] = cv2.perspectiveTransform(np.array([[[c[0][0], c[0][1]]]]), inv_trans)
    r[0] = cv2.perspectiveTransform(np.array([[[r[0][0], r[0][1]]]]), inv_trans)
    #aaa = matrix * np.array([l[0][0], l[0][1], 1])
    return l, c, r
    # cv2.imshow('left', left)
    # cv2.imshow('center', center)
    # cv2.imshow('right', right)
    # cv2.imshow('perspective', perspective)
    # cv2.imshow('original', frame)
    
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()