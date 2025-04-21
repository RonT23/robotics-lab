# Build and Run a Custom ROS Package 

## 1. Configure the System (for Linux)

```shell
sudo apt update
sudo apt install -y apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io
```

## 1. Configure the System (for Windows)

Open powershell as admin.

```shell
wsl --install
```

Once installed reboot the system. 

```shell
wsl --set-default-version 2
```

Then install Docker Desktop for Windows and from settings > resources
make sure that Ubuntu is ON.

Open the WSL terminal and install GUI support:

```shell
sudo apt update
sudp apt install x11-apps -y
export DISPLAY=:0
```

## 1. Build the ROS Noetic Docker Image

```shell
docker build -t ros-docker .
```

## 2. Start the Container 

Run the following commands on 2 terminals

```shell
chmod +x start-docker-image.sh
xhost +local:docker # export DISPLAY=:0 for Windows on WSL
./start-docker-image.sh
```

## 3. Compile the Package (terminal 1)

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

## 4. Run the Gazebo Simulator (terminal 2)

```shell
cd ~/catkin_ws
source devel/setup.bash
roslaunch xarm_gazebo zarm7_pf.launch
```

## 5. Start the Package (terminal 1)

```shell
roslaunch robosys_path_following path_following_3dof.launch
```
