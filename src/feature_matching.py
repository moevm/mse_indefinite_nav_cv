from datetime import datetime
import cv2
import numpy as np
from curve import get_line


class Matcher:
    def __init__(self, features='orb'):
        self.timesum = 0
        self.featsum = 0
        self.matchsum = 0
        self.frames = []
        self.last_H = np.identity(3)
        self.last_frame = []
        self.descs_prev = []
        self.kps_prev = []
        self.drawing = False

        # create detector and matcher
        if features == 'orb':
            self.detector = cv2.ORB_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        elif features == 'sift':
            self.detector = cv2.SIFT_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        else:
            print('Wrong featurer')

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

    def calculate_matches(self, im_prev, im_new, show=False):
        first = False

        if len(self.descs_prev) == 0:
            first = True

        #im_new = self._unsharp(im_new)
        im_new_bw = cv2.cvtColor(im_new, cv2.COLOR_BGR2GRAY)

        if first:
            #im_prev = self._unsharp(im_prev)
            im_prev_bw = cv2.cvtColor(im_prev, cv2.COLOR_BGR2GRAY)

        a = datetime.now()

        # detect
        if first:
            self.kps_prev, self.descs_prev = self.detector.detectAndCompute(im_prev_bw, None)
        im_new_kps, im_new_descs = self.detector.detectAndCompute(im_new_bw, None)

        # match and ratio test
        good_m = []
        if not(self.descs_prev is None) and not(im_new_descs is None):
            matches = self.matcher.knnMatch(
                self.descs_prev, im_new_descs, k=2)
            # print("Matches", len(matches))
            for m in matches:
                if len(m) > 1 and m[0].distance < 0.75 * m[1].distance:
                    good_m.append([m[0]])

        im_prev_kps = self.kps_prev
        self.kps_prev = im_new_kps
        self.descs_prev = im_new_descs

        b = datetime.now()

        # if show:
        #     self.draw_matches(im1, im2, queryKeypoints, trainKeypoints, good_m)
        #     self.draw_stitching(im1, im2, queryKeypoints,
        #                         trainKeypoints, good_m)

        # update stats
        self.featsum += len(im_new_kps)
        self.matchsum += len(good_m)
        self.timesum += (b - a).microseconds

        return im_prev_kps, im_new_kps, good_m

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

    def transform_pts(self, frame, descs_prev, pts, show=False):
        kpsA, kpsB, matches = self.calculate_matches(
            descs_prev, frame, show)
        H = self.get_homography(kpsA, kpsB, matches)
        pts = np.float32(pts).reshape(-1, 1, 2)
        if H is not None and len(H) > 0 and len(pts) > 0:
            return cv2.perspectiveTransform(pts, H)
        return pts
    
    def is_drawing(self):
        return self.drawing

    def is_in_frame(point, w, h):
        return point[0][0] > 0 and point[0][1] > 0 and point[0][0] < h and point[0][1] < w

    def frame_transform(self, frame, p_far=[], dir=0):
        if not self.drawing:
            if len(p_far) == 2:
                self.pts_l, self.pts_r = get_line(frame, p_far, dir)
                self.drawing = True
            else:
                self.last_frame = frame
                return frame
        else:
            points = self.pts_l
            if len(self.pts_r) > 0: 
                points = np.concatenate([self.pts_l, self.pts_r]) 
            transformed = self.transform_pts(
                frame, self.last_frame, points, False).squeeze()
            self.pts_l = transformed[0:len(self.pts_l)]
            if len(self.pts_r) > 0:
                self.pts_r = transformed[len(self.pts_l):]

        self.last_frame = frame

        h, w = frame.shape[:2]

        # delete out of frame points
        del_inds = []
        for i in range(len(self.pts_l)):
            p = self.pts_l[i]
            if p[0] < -100 or p[0] > w + 100 or p[1] < -100 or p[1] > h + 100:
                del_inds.append(i)
        if len(del_inds) > 0:
            self.pts_l = np.delete(self.pts_l, del_inds, 0)
            self.pts_l = np.float32(self.pts_l).reshape(-1, 1, 2).squeeze()

        del_inds = []
        for i in range(len(self.pts_r)):
            p = self.pts_r[i]
            if p[0] < -100 or p[0] > w + 100 or p[1] < -100 or p[1] > h + 100:
                del_inds.append(i)
        if len(del_inds) > 0:
            self.pts_r = np.delete(self.pts_r, del_inds, 0)
            self.pts_r = np.float32(self.pts_r).reshape(-1, 1, 2).squeeze()

        if len(self.pts_l) <= 2 or len(self.pts_r) <= 2:
            self.drawing = False
            return frame

        # draw lines
        cv2.polylines(frame, [np.array(self.pts_l, np.int32)],
                      False, [0, 255, 255], 5)
        if len(self.pts_r) > 0:
            cv2.polylines(frame, [np.array(self.pts_r, np.int32)],
                      False, [255, 255, 255], 5)
        return frame

    # def draw_transform(self, frames, p_near, p_far, dir, show=False):
    #     self.pts, self.col = get_line(frames[0], p_near, p_far, dir)

    #     for i in range(1, len(frames)):
    #         frame = frames[i]
    #         pts = self.transform_pts(frame, frames[i-1], pts, show)

    #         # check if all points are bad
    #         isok = False
    #         h, w = frame.shape[:2]
    #         for p in pts:
    #             p = p[0]
    #             if p[0] > 0 and p[0] < h and p[1] > 0 and p[1] < w:
    #                 isok = True
    #                 break
    #         if not isok:  # stop drawing line
    #             return i

    #         cv2.polylines(frame, [np.array(pts, np.int32)],
    #                       False, self.col, 3)
    #         self.frames.append(frame)
    #         # cv2.imshow("frame", frame)
    #         # cv2.waitKey(1)
    #     return len(frames)
