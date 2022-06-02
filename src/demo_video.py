import cv2
from feature_matching import Matcher
import template_matching
import numpy as np


def get_point_from_rect(p1, p2):
    return np.array([p1[0] + (p2[0] - p1[0]) / 2, p1[1] + (p2[1] - p1[1]) / 2], np.int32)


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


if __name__ == "__main__":
    frames, fps = video_to_frames("neptunus.mp4")
    frames = frames[235:700]
    new_frames = []
    matcher = Matcher('sift')

    l, c, r = template_matching.run(frames[0])
    p_far = np.squeeze(l)
    updated_frame = matcher.frame_transform(frames[0], p_far, -1)

    for frame in frames:
        if matcher.is_drawing():
            updated_frame = matcher.frame_transform(frame)

        if updated_frame is None or len(updated_frame) == 0:
            updated_frame = frame
        new_frames.append(frame)

    # write video
    if not new_frames is None and len(new_frames) > 0:
        out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(
            'M', 'J', 'P', 'G'), fps, (new_frames[0].shape[1], new_frames[0].shape[0]))
        for frame in new_frames:
            out.write(frame)
        out.release()
    # statistics
    matcher.get_avg(len(frames))
