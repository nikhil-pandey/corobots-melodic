cmake_minimum_required(VERSION 2.8.3)
project(corobot_follower)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  message_generation
  corobot_openpose
  std_msgs
  sensor_msgs
  nav_msgs
  geometry_msgs
  cv_bridge
  image_transport
)

add_message_files(
  FILES
  OpenPoseGesture.msg
  OpenPoseGestureList.msg
  HumanLocation.msg
  HumanLocationList.msg
)

add_service_files(
  FILES
  GesturesSrv.srv
  LocationSrv.srv
)

generate_messages(
  DEPENDENCIES
  corobot_openpose
  std_msgs
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime corobot_openpose
)