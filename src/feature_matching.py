from datetime import datetime
import time
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


class Macther:
    def __init__(self):
        self.timesum = 0
        self.featsum = 0
        self.matchsum = 0
        self.frames = []

    def get_avg(self, frames_num):
        print("Avg time (2 detections + matching):", self.timesum / frames_num)
        print("Avg features:", self.featsum / frames_num)
        print("Avg matches:", self.matchsum / frames_num)

    def get_frames(self):
        return self.frames

    def draw_matches(self, im1, im2, features='orb'):
        # crop 2/5 psrt from top
        h1, w1 = im1.shape[:2]
        im1 = im1[int(h1 / 2.5):, :]
        h2, w2 = im2.shape[:2]
        im2 = im2[int(h2 / 2.5):, :]

        # convert to grayscale
        query_img_bw = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        train_img_bw = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

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
        matches = []
        good = []
        if not(queryDescriptors is None) and not(trainDescriptors is None):
            matches = matcher.knnMatch(queryDescriptors, trainDescriptors, k=2)
            #print("Matches", len(matches))
            for m in matches:
                if len(m) > 1 and m[0].distance < 0.75 * m[1].distance:
                    good.append([m[0]])

        b = datetime.now()

        # draw macthes
        match_img = cv2.drawMatchesKnn(im1, queryKeypoints,
                                       im2, trainKeypoints, good, None)

        cv2.putText(match_img, "time: {} features1: {} features2 {} matches: {}".format(b - a, len(queryKeypoints), len(trainKeypoints), len(good)),
                    (0, 20),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    (255, 0, 0),
                    1,
                    1)

        # update stats
        self.featsum += len(queryKeypoints)
        self.matchsum += len(good)
        self.timesum += (b - a).microseconds

        # stitching
        kpsA = np.float32([kp.pt for kp in queryKeypoints])
        kpsB = np.float32([kp.pt for kp in trainKeypoints])

        ptsA = []
        ptsB = []

        h1, w1 = im1.shape[:2]
        h2, w2 = im2.shape[:2]

        if len(good) > 4:
            # construct the two sets of points
            for m in good:
                ptsA.append(kpsA[m[0].queryIdx])
                ptsB.append(kpsB[m[0].trainIdx])

            ptsA = np.float32(ptsA)
            ptsB = np.float32(ptsB)

            # find transform matrix for second image
            H = cv2.findHomography(ptsB, ptsA, cv2.RANSAC)[0]

            final_im2 = np.ndarray((h1+h2, w1+w2, 3), dtype=np.uint8)
            final_im2[:, :] = [0, 0, 0]
            final_im2[:h1, :w1] = im2

            if not H is None:
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
                    im1, queryKeypoints, None, color=[255, 0, 0])
                im2 = cv2.drawKeypoints(
                    im2, trainKeypoints, None, color=[0, 0, 255])

                # transform second image
                final_im2 = cv2.warpPerspective(
                    im2, H, (int(w1 + w2), int(h1 + h2)))

            big_im1 = np.ndarray((h1+h2, w1+w2, 3), dtype=np.uint8)
            big_im1[:, :] = [0, 0, 0]
            big_im1[:h1, :w1] = im1

            # blend im1 and im2 with transparency
            final_img = cv2.addWeighted(final_im2, 0.5, big_im1, 0.5, 0)

            # show the final images
            self.frames.append(final_img)
            #cv2.imshow("Stitching", final_img)
            #cv2.imshow("Matches", match_img)
            # cv2.waitKey(1)


if __name__ == "__main__":
    frames, fps = video_to_frames("./perekrestok.mp4")
    m = Macther()
    for i in range(1, len(frames)):
        m.draw_matches(frames[i - 1], frames[i], 'orb')
    new_frames = m.get_frames()
    # write video
    if not new_frames is None and len(new_frames) > 0:
        out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(
            'M', 'J', 'P', 'G'), fps, (new_frames[0].shape[1], new_frames[0].shape[0]))
        for frame in new_frames:
            out.write(frame)
        out.release()
    # statistics
    m.get_avg(len(frames))
