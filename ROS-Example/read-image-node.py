#!/usr/bin/env python3

import os
import rospy

from duckietown.dtros import DTROS, NodeType
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage

class MySubscriberNode(DTROS):

    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(MySubscriberNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        # construct subscriber
        self.sub_compressed_image = rospy.Subscriber('~image_reading', CompressedImage, self.callback)
        
    def callback(data):
		rospy.loginfo("I receive compressed image!")


if __name__ == '__main__':
    # create the node
    node = MySubscriberNode(node_name='my_subscriber_node')
    # keep spinning
    rospy.spin()
