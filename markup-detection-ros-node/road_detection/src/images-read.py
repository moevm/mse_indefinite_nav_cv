#!/usr/bin/env python3

import rospy
import numpy as np
from duckietown.dtros import DTROS, NodeType
from std_msgs.msg import String, Header
from sensor_msgs.msg import CompressedImage
import duckietown_code_utils as dtu
from turbojpeg import TurboJPEG, TJPF_GRAY

import detection

class ImagesReader(DTROS):
    autobot_name = "autobot05"

    def __init__(self, node_name):
        super(ImagesReader, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self.history = np.array([[0, 0, 0, 0]])
        self.sub = rospy.Subscriber(f'/{autobot_name}/camera_node/image/compressed', CompressedImage, self.callback)
        self.marked_pub = rospy.Publisher(f'/{autobot_name}/marked/roads/image/compressed', CompressedImage, None)
        self._jpeg = TurboJPEG()

    def callback(self, ros_frame):
        rospy.loginfo(f"Got video message. header.frame_id={ros_frame.header.frame_id}, format={ros_frame.format}, data.len={len(ros_frame.data)}")

        image_cv = dtu.bgr_from_jpg(ros_frame.data)
        updated_frame, process_history = detection.process_frame(image_cv, self.history) # get frame with detected road markings
        self.history = process_history
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
