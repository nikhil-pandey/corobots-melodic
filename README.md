## Introduction


## Installation


## Running

### On Corobot
```bash
mkdir corobot_ws
catkin init
git clone git@github.com:nikhil-pandey/corobots-melodic.git src
catkin config --blacklist openpose_ros
catkin build
source devel/setup.bash
roslaunch corobot_bringup robot_base.launch
```

### On GPU machine
```bash
mkdir corobot_ws
git clone git@github.com:nikhil-pandey/corobots-melodic.git src
export ROS_MASTER_URL=http://129.21.84.118:11311
```