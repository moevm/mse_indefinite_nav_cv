#!/usr/bin/env python3

import rospy
import numpy as np
import cv2
from duckietown.dtros import DTROS, NodeType
from std_msgs.msg import String, Header
from sensor_msgs.msg import CompressedImage
import duckietown_code_utils as dtu
from turbojpeg import TurboJPEG, TJPF_GRAY

#import detection

class ImagesReader(DTROS):

    def __init__(self, node_name):
        autobot_name = "autobot07"
        super(ImagesReader, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self.sub = rospy.Subscriber(f'/{autobot_name}/camera_node/image/compressed', CompressedImage, self.callback)
        self.marked_pub = rospy.Publisher(f'/{autobot_name}/marked/roads/image/compressed', CompressedImage, None)
        self._jpeg = TurboJPEG()

    def callback(self, ros_frame):
        color_1 = (255, 0, 0)
        color_2 = (0, 255, 0)
        color_3 = (0, 0, 255)
        thickness = 3
        area_1 = [(35, 130), (150, 250)]
        area_2 = [(200, 150), (375, 200)]
        area_3 = [(475, 175), (638, 330)]

        rospy.loginfo(f"Got video message. header.frame_id={ros_frame.header.frame_id}, format={ros_frame.format}, data.len={len(ros_frame.data)}")

        image_cv = dtu.bgr_from_jpg(ros_frame.data)
        updated_frame = cv2.rectangle(image_cv, area_1[0], area_1[1], color_1, thickness)
        updated_frame = cv2.rectangle(updated_frame, area_2[0], area_2[1], color_2, thickness)
        updated_frame = cv2.rectangle(updated_frame, area_3[0], area_3[1], color_3, thickness)
        self._publish_frame(ros_frame, updated_frame)

    def _publish_frame(self, ros_frame, updated_frame):
        pub_msg = CompressedImage()
        pub_msg.header.seq = ros_frame.header.seq
        pub_msg.header.stamp = ros_frame.header.stamp
        pub_msg.header.frame_id = ros_frame.header.frame_id
        pub_msg.format = "jpeg"
        pub_msg.data = self._jpeg.encode(updated_frame)

        rospy.loginfo(f"Trying to send frame. header.frame_id={pub_msg.header.frame_id}, format={pub_msg.format}, data.len={len(pub_msg.data)}")
        self.marked_pub.publish(pub_msg)

if __name__ == '__main__':
    node = ImagesReader(node_name='images-read')
    rospy.spin()