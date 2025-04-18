cmake_minimum_required(VERSION 2.8.3)
project(robosys_path_following)

## Find catkin and any catkin packages
find_package(catkin REQUIRED COMPONENTS roscpp rospy std_msgs geometry_msgs message_generation genmsg)

## Declare ROS messages and services
#add_message_files(FILES)
#add_service_files(FILES)

## Generate added messages and services
generate_messages(DEPENDENCIES std_msgs geometry_msgs)

## Declare a catkin package
catkin_package(CATKIN_DEPENDS message_runtime)

## Build follower
include_directories(include ${catkin_INCLUDE_DIRS})

#add_executable(main src/main.cpp)
#target_link_libraries(main ${catkin_LIBRARIES})
#add_dependencies(main robosys_path_following)
#add_dependencies(main ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
