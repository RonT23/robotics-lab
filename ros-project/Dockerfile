FROM ros:noetic

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_WORKSPACE=/root/catkin_ws

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-yaml \
    git \
    wget \
    curl \
    ros-noetic-moveit \
    ros-noetic-moveit-visual-tools \
    ros-noetic-ros-control \
    ros-noetic-ros-controllers \
    ros-noetic-gazebo-ros-pkgs \
    ros-noetic-gazebo-ros-control \
    ros-noetic-joint-state-publisher \
    ros-noetic-robot-state-publisher \
    ros-noetic-tf \
    ros-noetic-rviz \
    ros-noetic-sensor-msgs \
    ros-noetic-std-msgs \
    ros-noetic-rospy \
    python3-catkin-tools \
    tmux \
    x11-apps && \
    rm -rf /var/lib/apt/lists/*

# Python ROS packages
RUN pip3 install rospkg catkin_pkg
RUN pip3 matplotlib

# Setup ROS environment
RUN rosdep init && rosdep update
RUN rm -f /etc/ros/rosdep/sources.list.d/20-default.list && \
    rosdep init && \
    rosdep update

# Create catkin workspace
RUN mkdir -p ${ROS_WORKSPACE}/src && \
    cd ${ROS_WORKSPACE} && \
    catkin init

# Clone required package
RUN cd ${ROS_WORKSPACE}/src && \
    git clone https://github.com/oikonpar/xarm_ros.git

# Patch CMakeLists.txt for Noetic (line 223)
RUN sed -i '223s/${catkin_LIBRARIES}/${CATKIN_PACKAGE_BIN_DESTINATION}/' \
    ${ROS_WORKSPACE}/src/xarm_ros/xarm_gazebo/CMakeLists.txt || true

# Build workspace
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && \
    cd ${ROS_WORKSPACE} && \
    rosdep install --from-paths src --ignore-src -y --rosdistro noetic && \
    catkin_make"

# Source on startup
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc && \
    echo "source ${ROS_WORKSPACE}/devel/setup.bash" >> ~/.bashrc

WORKDIR ${ROS_WORKSPACE}
CMD ["bash"]
