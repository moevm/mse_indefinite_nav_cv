#!/usr/bin/env python3

import os
import rospy
from duckietown.dtros import DTROS, NodeType
from std_msgs.msg import String, Header
from sensor_msgs.msg import CompressedImage

import detection

class ImagesReader(DTROS):
    def __init__(self, node_name):
        super(ImagesReader, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self.history = None
        self.sub = rospy.Subscriber('/autobot10/camera_node/image/compressed', CompressedImage, self.callback)

    def callback(self, frame_info):
        rospy.loginfo(f"Got video message. header.frame_id={frame_info.header.frame_id}, format={frame_info.format}, data.len={len(frame_info.data)}")
        updated_frame, process_history = detection.process_frame(list(frame_info.data), self.history) # get frame with detected road markings
        self.history = process_history
        self._publish_frame_to_backfile(updated_frame)

    def _publish_frame_to_backfile(self, updated_frame):
        if updated_frame:
            print(f"Got detection-frame")

if __name__ == '__main__':
    node = ImagesReader(node_name='images-read')
    rospy.spin()