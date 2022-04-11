from datetime import datetime
import time
import cv2


def video_to_frames(path):
    frames_list = []
    videoCapture = cv2.VideoCapture(path)
    frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    for i in range(int(frames)):
        ret, frame = videoCapture.read()
        frames_list.append(frame)
    return frames_list


class Macther:
    def __init__(self):
        self.timesum = 0
        self.featsum = 0
        self.matchsum = 0

    def get_avg(self, frames_num):
        print("Avg time (2 detections + matching):", self.timesum / frames_num)
        print("Avg features:", self.featsum / frames_num)
        print("Avg matches:", self.matchsum / frames_num)

    def draw_matches(self, im1, im2):
        # Convert it to grayscale
        h1 = im1.shape[0]
        im1 = im1[int(h1 / 3):, :]
        h2 = im2.shape[0]
        im2 = im2[int(h2 / 3):, :]

        query_img_bw = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        train_img_bw = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

        matcher = cv2.SIFT_create()
        #matcher = cv2.ORB_create()

        # Now detect the keypoints and compute
        # the descriptors for the query image
        # and train image
        a = datetime.now()
        queryKeypoints, queryDescriptors = matcher.detectAndCompute(
            query_img_bw, None)
        trainKeypoints, trainDescriptors = matcher.detectAndCompute(
            train_img_bw, None)

        # Initialize the Matcher for matching
        # the keypoints and then match the
        # keypoints
        matcher = cv2.BFMatcher()
        matches = []
        good = []
        if not(trainDescriptors is None) and not(trainDescriptors is None):
            matches = matcher.knnMatch(queryDescriptors, trainDescriptors, k=2)
            if len(matches) > 10:
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good.append([m])

        b = datetime.now()
        
        final_img = cv2.drawMatchesKnn(im1, queryKeypoints,
                                       im2, trainKeypoints, good, None)

        font = cv2.FONT_HERSHEY_PLAIN
        bottomLeftCornerOfText = (0, 20)
        fontScale = 1
        fontColor = (255, 0, 0)
        thickness = 1
        lineType = 1

        cv2.putText(final_img, "time: {} features1: {} features2 {} matches: {}".format(b - a, len(queryKeypoints), len(trainKeypoints), len(good)),
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)
        self.featsum += len(queryKeypoints)
        self.matchsum += len(good)
        self.timesum += (b - a).microseconds
        # Show the final image
        cv2.imshow("Matches", final_img)
        cv2.waitKey(10)


if __name__ == "__main__":
    frames = video_to_frames("./perekrestok.mp4")
    m = Macther()
    for i in range(1, len(frames)):
        m.draw_matches(frames[i - 1], frames[i])
    m.get_avg(len(frames))
