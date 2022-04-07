from datetime import datetime
import time
import cv2


def video_to_frames(path):
    frames_list = []
    videoCapture = cv2.VideoCapture(path)
    frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    videoCapture.open(path)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    for i in range(int(frames)):
        ret, frame = videoCapture.read()
        frames_list.append(frame)
    return frames_list


def draw_matches(im1, im2):
    # Convert it to grayscale
    query_img_bw = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    train_img_bw = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Initialize the ORB detector algorithm
    orb = cv2.ORB_create()

    # Now detect the keypoints and compute
    # the descriptors for the query image
    # and train image
    a = datetime.now()
    queryKeypoints, queryDescriptors = orb.detectAndCompute(query_img_bw, None)
    trainKeypoints, trainDescriptors = orb.detectAndCompute(train_img_bw, None)

    # Initialize the Matcher for matching
    # the keypoints and then match the
    # keypoints
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(queryDescriptors, trainDescriptors, k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    b = datetime.now()
    # draw the matches to the final image
    # containing both the images the drawMatches()
    # function takes both images and keypoints
    # and outputs the matched query image with
    # its train image
    final_img = cv2.drawMatchesKnn(im1, queryKeypoints,
                                im2, trainKeypoints, good, None)

    #final_img = cv2.resize(final_img, (1000, 650))

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
    # Show the final image
    cv2.imshow("Matches", final_img)
    cv2.waitKey(300)


if __name__ == "__main__":
    frames = video_to_frames("../video.mp4")
    for i in range(1, len(frames)):
        draw_matches(frames[i - 1], frames[i])
