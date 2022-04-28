import cv2
import numpy as np


def gen_bezier_quadratic(p0, p1, p2, num=50):
  t_ar = np.linspace(0, 1, num)
  res = [(1 - t)**2 * p0 + 2 * t * (1 - t) * p1 + t**2 * p2 for t in t_ar]
  return res


def draw_line(img, p_near, p_far, dir):
    # filter
    new = cv2.GaussianBlur(img, (5, 5), 5)
    cv2.addWeighted(img, 1.5, new, -0.5, 0, new)

    if not dir is None:
        if np.abs(dir) == 1: # left or right
          col = []
          if (dir == -1): # left
            col = [0, 0, 0] # right line
          else: # right
            col = [255, 0, 255] # left line
          p_corner = np.array([p_near[0], p_far[1]])
          pts = np.array(gen_bezier_quadratic(p_near, p_corner, p_far), np.int32)
          pts = pts.reshape((-1, 1, 2))
          cv2.polylines(img, [pts], False, [0, 0, 0], 3)
        # if dir == 0: # forward
        #   cv2.line(img, p_near, p_far, [0, 0, 0], 3)
    return img


if __name__ == "__main__":
    img = cv2.imread("../perekrestok.png")
    img_new = draw_line(img, np.array([590, 358]), np.array([77, 375]), np.array([146, 190]))
    cv2.imshow("Liniya", img_new)
    cv2.waitKey(10000)
