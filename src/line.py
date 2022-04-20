import cv2
import numpy as np


def gen_bezier_quadratic(p0, p1, p2, num=50):
  t_ar = np.linspace(0, 1, num)
  res = [(1 - t)**2 * p0 + 2 * t * (1 - t) * p1 + t**2 * p2 for t in t_ar]
  return res


def draw_line(img, p_near, p_far):
    p_corner = np.array([p_near[0], p_far[1]])
    pts = np.array(gen_bezier_quadratic(p_near, p_corner, p_far), np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(img, [pts], False, [255, 0, 0], 3)
    cv2.imshow("Liniya", img)
    cv2.waitKey(10000)


if __name__ == "__main__":
    img = cv2.imread("../perekrestok.png")
    draw_line(img, np.array([590, 358]), np.array([146, 190]))
