cmake_minimum_required(VERSION 2.8.3)
project(camera_display)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    sensor_msgs
    cv_bridge
)

find_package(OpenCV REQUIRED)

catkin_package()

include_directories(
    ${catkin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(camera_display src/camera_display.cpp)

target_link_libraries(camera_display
    ${catkin_LIBRARIES}
    ${OpenCV_LIBRARIES}
)

