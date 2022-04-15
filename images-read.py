#!/usr/bin/env python3

import os
import rospy
import numpy as np
from duckietown.dtros import DTROS, NodeType
from std_msgs.msg import String, Header
from sensor_msgs.msg import CompressedImage
import duckietown_code_utils as dtu
import duckietown_rosdata_utils as dru
from turbojpeg import TurboJPEG, TJPF_GRAY

import detection

class ImagesReader(DTROS):
    def __init__(self, node_name):
        super(ImagesReader, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self.history = np.array([[0, 0, 0, 0]])
        self.sub = rospy.Subscriber('/autobot05/camera_node/image/compressed', CompressedImage, self.callback)
        self.marked_pub = rospy.Publisher('/autobot05/marked/image/compressed', CompressedImage, None)
        self._jpeg = TurboJPEG()

    def callback(self, frame_info):
        rospy.loginfo(f"Got video message. header.frame_id={frame_info.header.frame_id}, format={frame_info.format}, data.len={len(frame_info.data)}")
        img = dtu.bgr_from_rgb(dru.rgb_from_ros(frame_info))
        updated_frame, process_history = detection.process_frame(img, self.history) # get frame with detected road markings
        self.history = process_history
        self._publish_frame(frame_info, updated_frame)

    def _publish_frame(self, frame_info, updated_frame):
        print(f"{type(updated_frame)} and {updated_frame.shape}")

        pub_msg = CompressedImage()
        pub_msg.header.seq = frame_info.header.seq
        pub_msg.header.stamp = frame_info.header.stamp
        pub_msg.header.frame_id = frame_info.header.frame_id
        pub_msg.format = "jpeg"
        pub_msg.data = self._jpeg.encode(updated_frame)

        rospy.loginfo(f"Try to send frame. header.frame_id={pub_msg.header.frame_id}, format={pub_msg.format}, data.len={len(pub_msg.data)}")
        self.marked_pub.publish(frame_info)

if __name__ == '__main__':
    node = ImagesReader(node_name='images-read')
    rospy.spin()
