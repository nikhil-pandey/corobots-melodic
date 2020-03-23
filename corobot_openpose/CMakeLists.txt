cmake_minimum_required(VERSION 2.8.3)
project(corobot_openpose)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  message_generation
  cv_bridge
  image_transport
)

add_message_files(
  FILES
  BoundingBox.msg
  OpenPoseHuman.msg
  OpenPoseHumanList.msg
  PointWithProb.msg
)

add_service_files(
  FILES
  EstimatePoseSrv.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime
)