import cv2


def unsharp(img):
    new = cv2.GaussianBlur(img, (0, 0), 9)
    cv2.addWeighted(img, 1.5, new, -0.5, 0, new)
    cv2.imshow("Old", img)
    cv2.imshow("New", new)
    cv2.waitKey(10000)


if __name__ == "__main__":
    img = cv2.imread("../perekrestok.png")
    unsharp(img)
