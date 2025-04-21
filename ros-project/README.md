# Build and Run a Custom ROS Package 

## 1. Build the ROS Noetic Docker Image

```shell
docker build -t ros-docker .
```

## 2. Make a Shared Folder

```shell
mkdir -p ~/ros-docker/external
```

## 3. Copy the Provided Package

```shell
cp -r robosys_path_following ~/ros-docker/external
```

## 4. Start the Container 

Run the following commands on 2 terminals

```shell
chmod +x start-docker-image.sh
./start-docker-image.sh
```

## 5. Compile the Package

On terminal 1:

```shell
cd ~/catkin_ws
source devel/setup.bash
rosdep install --from-paths src/external --ignore-src -r -y
catkin_make
cd ~/catkin_ws/src/external/robosys_path_following/scripts
chmod +x controller_3dof.py 
chmod +x controller_7dof.py
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```
## 6. Run the Gazebo Simulator

On terminal 2:

```shell
cd ~/catkin_ws
source devel/setup.bash
roslaunch xarm_gazebo xarm7_pf.launch
```

## 7. Start the Package

On terminal 1:

```shell
roslaunch robosys_path_following path_following_3dof.launch
```
