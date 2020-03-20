## Introduction
Uses camera and 2D lidar to follow a person and follow gestures.

## Installation
- Install [ROS Melodic](http://wiki.ros.org/melodic)
- Install [Openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)

### On Corobot
```bash
sudo apt-get install python-rosdep python-catkin-tools
mkdir -p corobot_ws/src
cd corobot_ws
catkin init
git clone git@github.com:nikhil-pandey/corobots-melodic.git src
git submodule update
rosdep update
rosdep install --from-paths src -i
catkin build
source devel/setup.bash
sudo usermod -a -G dialout $USER
roslaunch corobot_bringup robot_base.launch
```

### On GPU machine
```bash
export COROBOT_IP=<robot_ip_address>
export ROS_MASTER_URL=http://COROBOT_IP:11311
sudo apt-get install python-rosdep python-catkin-tools
mkdir -p corobot_ws/src
cd corobot_ws
git clone git@github.com:nikhil-pandey/corobots-melodic.git src
git submodule update
catkin build
source devel/setup.bash
roslaunch corobot_bringup gpu.launch
```