cmake_minimum_required(VERSION 2.8.3)
project(pcl_exercise)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  pcl_conversions
  pcl_ros
  roscpp
  rospy
  sensor_msgs
  std_msgs
  message_generation
  ddynamic_reconfigure  # realsense2
  bb_pointnet
  visualization_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_service_files(
  FILES
  norm.srv
)
generate_messages(
  DEPENDENCIES
  std_msgs  # Or other packages containing msgs
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime 
)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

add_executable(depth_to_point  src/depth_to_point.h src/depth_to_point.cpp)
target_link_libraries(depth_to_point ${catkin_LIBRARIES})
add_dependencies(depth_to_point src/depth_to_point.cpp ${catkin_EXPORTED_TARGETS})

add_executable(normal src/normal.cpp)
target_link_libraries(normal ${catkin_LIBRARIES})
add_dependencies(normal pcl_exercise_gencpp)

add_executable(depth_to_point_whole  src/depth_to_point_whole.h src/depth_to_point_whole.cpp)
target_link_libraries(depth_to_point_whole ${catkin_LIBRARIES})
add_dependencies(depth_to_point_whole src/depth_to_point_whole.cpp ${catkin_EXPORTED_TARGETS})

add_executable(pc_preprocessing src/pc_preprocessing.h src/pc_preprocessing.cpp)
target_link_libraries(pc_preprocessing ${catkin_LIBRARIES})
add_dependencies(pc_preprocessing src/pc_preprocessing.cpp ${catkin_EXPORTED_TARGETS})

add_executable(publish_marker src/publish_marker.h src/publish_marker.cpp)
target_link_libraries(publish_marker ${catkin_LIBRARIES})
add_dependencies(publish_marker src/publish_marker.cpp ${catkin_EXPORTED_TARGETS})

add_executable(mask_to_point src/mask_to_point.h src/mask_to_point.cpp)
target_link_libraries(mask_to_point ${catkin_LIBRARIES})
add_dependencies(mask_to_point src/mask_to_point.cpp ${catkin_EXPORTED_TARGETS})

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
