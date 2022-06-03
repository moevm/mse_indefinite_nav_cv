import cv2
import numpy as np


def gen_bezier_quadratic(p0, p1, p2, num=50):
    t_ar = np.linspace(0, 1, num)
    res = [(1 - t)**2 * p0 + 2 * t * (1 - t) * p1 + t**2 * p2 for t in t_ar]
    return res


def get_line(img, p_far, dir):
    pts = []
    col = []

    # experimental for 640x480
    p_near_l = np.array([150, 440]) #50
    p_near_r = np.array([590, 440])

    if not dir is None:
        if np.abs(dir) == 1:  # left or right
            p_far_l = []
            p_far_r = []

            if dir == -1:
                p_far_l = p_far
                p_far_r = np.array([p_far[0], p_far[1] - 50]) ###
            if dir == 1:
                p_far_l = p_far

            p_corner_l = np.array([p_near_l[0], p_far_l[1]])

            pts_l = np.array(gen_bezier_quadratic(
                p_near_l, p_corner_l, p_far_l), np.int32).reshape((-1, 1, 2)).squeeze()

            if dir == 1:
                pts_r = []
            else:
                p_corner_r = np.array([p_near_r[0], p_far_r[1]])
                pts_r = np.array(gen_bezier_quadratic(
                    p_near_r, p_corner_r, p_far_r), np.int32).reshape((-1, 1, 2)).squeeze()
            # cv2.polylines(img, [pts], False, [0, 0, 0], 3)
        # if dir == 0:  # forward
        #     pts_l = np.array([[p_near, p_far]])
        #     x_right = p_far[0]
        #     pts_l = np.array([[p_near, []])
        #   cv2.line(img, p_near, p_far, [0, 0, 0], 3)
    return pts_l, pts_r
