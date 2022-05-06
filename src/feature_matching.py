from datetime import datetime
import cv2
import numpy as np


def video_to_frames(path):
    frames_list = []
    videoCapture = cv2.VideoCapture(path)
    frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    for i in range(int(frames)):
        ret, frame = videoCapture.read()
        frames_list.append(frame)
    return frames_list, fps


def gen_bezier_quadratic(p0, p1, p2, num=50):
    t_ar = np.linspace(0, 1, num)
    res = [(1 - t)**2 * p0 + 2 * t * (1 - t) * p1 + t**2 * p2 for t in t_ar]
    return res


def get_line(img, p_near, p_far, dir):
    pts = []
    col = []
    if not dir is None:
        if np.abs(dir) == 1:  # left or right
            if (dir == -1):  # left
                col = [0, 0, 0]  # right line
            else:  # right
                col = [0, 255, 255]  # left line
            p_corner = np.array([p_near[0], p_far[1]])
            pts = np.array(gen_bezier_quadratic(
                p_near, p_corner, p_far), np.int32)
            pts = pts.reshape((-1, 1, 2))
            #cv2.polylines(img, [pts], False, [0, 0, 0], 3)
        # if dir == 0: # forward
        #   cv2.line(img, p_near, p_far, [0, 0, 0], 3)
    return pts, col


class Macther:
    def __init__(self):
        self.timesum = 0
        self.featsum = 0
        self.matchsum = 0
        self.frames = []
        self.last_H = np.identity(3)

    def get_avg(self, frames_num):
        print("Avg time (2 detections + matching):", self.timesum / frames_num)
        print("Avg features:", self.featsum / frames_num)
        print("Avg matches:", self.matchsum / frames_num)

    def get_frames(self):
        return self.frames

    def _unsharp(self, img):
        new = cv2.GaussianBlur(img, (3, 3), 9)
        cv2.addWeighted(img, 1.5, new, -0.5, 0, new)
        return img

    def calculate_matches(self, im1, im2, features='orb', show=False):
        # crop 4/10 part from top
        h1, w1 = im1.shape[:2]
        query_img = im1  # [int(h1 * 0.3):, :]
        h2, w2 = im2.shape[:2]
        train_img = im2  # [int(h2 * 0.3):, :]

        # filter
        query_img = self._unsharp(query_img)
        train_img = self._unsharp(train_img)

        # convert to grayscale
        query_img_bw = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
        train_img_bw = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)

        # create detector and matcher
        if features == 'orb':
            detector = cv2.ORB_create()
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        elif features == 'sift':
            detector = cv2.SIFT_create()
            matcher = cv2.BFMatcher(cv2.NORM_L2)
        else:
            print('Wrong featurer')
            return

        a = datetime.now()

        # detect
        queryKeypoints, queryDescriptors = detector.detectAndCompute(
            query_img_bw, None)
        trainKeypoints, trainDescriptors = detector.detectAndCompute(
            train_img_bw, None)

        # match and ratio test
        good_m = []
        if not(queryDescriptors is None) and not(trainDescriptors is None):
            matches = matcher.knnMatch(queryDescriptors, trainDescriptors, k=2)
            # print("Matches", len(matches))
            for m in matches:
                if len(m) > 1 and m[0].distance < 0.75 * m[1].distance:
                    good_m.append([m[0]])

        b = datetime.now()

        if show:
            self.draw_matches(im1, im2, queryKeypoints, trainKeypoints, good_m)
            self.draw_stitching(im1, im2, queryKeypoints, trainKeypoints, good_m)

        # update stats
        self.featsum += len(queryKeypoints)
        self.matchsum += len(good_m)
        self.timesum += (b - a).microseconds

        return queryKeypoints, trainKeypoints, good_m

    def draw_stitching(self, im1, im2, kpsA, kpsB, matches):
        h1, w1 = im1.shape[:2]
        h2, w2 = im2.shape[:2]

        final_im2 = np.ndarray((h1+h2, w1+w2, 3), dtype=np.uint8)
        final_im2[:, :] = [0, 0, 0]
        final_im2[:h1, :w1] = im2
        H = self.get_homography(kpsA, kpsB, matches)

        if not H is None and len(H) > 0:
            # borders
            im1[0, :] = [255, 0, 0]
            im1[:, 0] = [255, 0, 0]
            im1[h1-1, :] = [255, 0, 0]
            im1[:, w1-1] = [255, 0, 0]

            im2[0, :] = [0, 0, 255]
            im2[:, 0] = [0, 0, 255]
            im2[h2-1, :] = [0, 0, 255]
            im2[:, w2-1] = [0, 0, 255]

            # draw keypoints on frames
            im1 = cv2.drawKeypoints(
                im1, kpsA, None, color=[255, 0, 0])
            im2 = cv2.drawKeypoints(
                im2, kpsB, None, color=[0, 0, 255])

            # transform second image (+ scale for drawing)
            final_im2 = cv2.warpPerspective(
                im2, H, (int(w1 + w2), int(h1 + h2)))

        # scale im1
        big_im1 = np.ndarray((h1+h2, w1+w2, 3), dtype=np.uint8)
        big_im1[:, :] = [0, 0, 0]
        big_im1[:h1, :w1] = im1

        # blend im1 and im2 with transparency
        final_img = cv2.addWeighted(final_im2, 0.5, big_im1, 0.5, 0)

        # show the final images
        cv2.imshow("Stitching", final_img)
        cv2.waitKey(1)

    def get_homography(self, kpsA, kpsB, good_m):
        kpsA = np.float32([kp.pt for kp in kpsA])
        kpsB = np.float32([kp.pt for kp in kpsB])

        ptsA = []
        ptsB = []

        if len(good_m) >= 4:
            # construct the two sets of points
            for m in good_m:
                ptsA.append(kpsA[m[0].queryIdx])
                ptsB.append(kpsB[m[0].trainIdx])

            ptsA = np.float32(ptsA)
            ptsB = np.float32(ptsB)

            # find transform matrix for second image
            self.last_H = cv2.findHomography(ptsA, ptsB, cv2.RANSAC)[0]
        return self.last_H

    def draw_matches(self, im1, im2, kp1, kp2, matches):
        match_img = cv2.drawMatchesKnn(im1, kp1,
                                       im2, kp2, matches, None)

        cv2.putText(match_img,
                    "features1: {} features2 {} matches: {}".format(
                        len(kp1), len(kp2), len(matches)),
                    (0, 20),
                    cv2.FONT_HERSHEY_PLAIN, 1,
                    (255, 0, 0), 1, 1)
        cv2.imshow("Matches", match_img)
        cv2.waitKey(1)

    def transform_pts(self, frame, frame_prev, pts, show=False):
        kpsA, kpsB, matches = self.calculate_matches(
            frame_prev, frame, 'sift', show)
        H = self.get_homography(kpsA, kpsB, matches)
        if H is not None and len(H) > 0:
            pts = np.float32(pts).reshape(-1, 1, 2)
            return cv2.perspectiveTransform(pts, H)
        return pts

    def draw_transform(self, frames, p_near, p_far, dir, show=False):
        pts, col = get_line(frames[0], p_near, p_far, dir)

        for i in range(1, len(frames)):
            frame = frames[i]
            pts = self.transform_pts(frame, frames[i-1], pts, show)

            # check if all points are bad
            isok = False
            h, w = frame.shape[:2]
            for p in pts:
                p = p[0]
                if p[0] > 0 and p[0] < h and p[1] > 0 and p[1] < w:
                    isok = True
                    break
            if not isok:  # stop drawing line
                return i

            cv2.polylines(frame, [np.array(pts, np.int32)],
                          False, col, 3)
            self.frames.append(frame)
            # cv2.imshow("frame", frame)
            # cv2.waitKey(1)
        return len(frames)

def get_point_from_rect(p1, p2):
    return np.array([p1[0] + (p2[0] - p1[0]) / 2, p1[1] + (p2[1] - p1[1]) / 2], np.int32)

if __name__ == "__main__":
    frames, fps = video_to_frames("./tesla.mp4")

    # for i in range(len(frames)):
    #     h1, w1 = frames[i].shape[:2]
    #     frames[i] = frames[i][int(h1 * 0.3):, :]

    p_start = np.array([90, 500], np.int32)#get_point_from_rect([70, 440], [110, 460])
    p_end = np.array([580, 230], np.int32)#get_point_from_rect([480, 190], [520, 210])
    frames = frames[850:1000]

    m = Macther()
    last = m.draw_transform(frames, p_start, p_end, 1, True)

    # write video
    new_frames = m.get_frames()
    if (last < len(frames)):
        new_frames.extend(frames[last:])
    if not new_frames is None and len(new_frames) > 0:
        out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(
            'M', 'J', 'P', 'G'), fps, (new_frames[0].shape[1], new_frames[0].shape[0]))
        for frame in new_frames:
            out.write(frame)
        out.release()
    # statistics
    m.get_avg(len(frames))
