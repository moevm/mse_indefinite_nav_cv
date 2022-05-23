cmake_minimum_required(VERSION 2.8.3)
project(defining_markup_beginning_area)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  duckietown_msgs # Every duckietown packages should use this.
)

catkin_package()
