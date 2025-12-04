cmake_minimum_required(VERSION 3.14)
project(mujoco_ros)

# C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(mujoco_ros_msgs REQUIRED)

# Find MuJoCo manually
# Use MUJOCO_ROOT_DIR environment variable
set(MUJOCO_ROOT_DIR $ENV{MUJOCO_ROOT_DIR})
if(NOT MUJOCO_ROOT_DIR)
    set(MUJOCO_ROOT_DIR $ENV{MUJOCO_PATH})
endif()

message(STATUS "Looking for MuJoCo in: ${MUJOCO_ROOT_DIR}")

find_path(MUJOCO_INCLUDE_DIR
    NAMES mujoco/mujoco.h
    PATHS
        ${MUJOCO_ROOT_DIR}/include
        /usr/local/include
        /usr/include
        ~/.mujoco/mujoco-3.1.3/include
        ~/.mujoco/mujoco-3.0.0/include
    NO_DEFAULT_PATH
)

find_library(MUJOCO_LIBRARY
    NAMES mujoco
    PATHS
        ${MUJOCO_ROOT_DIR}/lib
        ${MUJOCO_ROOT_DIR}/lib64
        /usr/local/lib
        /usr/local/lib64
        /usr/lib
        /usr/lib64
        ~/.mujoco/mujoco-3.1.3/lib
        ~/.mujoco/mujoco-3.0.0/lib
    NO_DEFAULT_PATH
)

if(MUJOCO_INCLUDE_DIR AND MUJOCO_LIBRARY)
    message(STATUS "Found MuJoCo include: ${MUJOCO_INCLUDE_DIR}")
    message(STATUS "Found MuJoCo library: ${MUJOCO_LIBRARY}")
else()
    message(FATAL_ERROR "MuJoCo not found. Please set MUJOCO_ROOT_DIR environment variable.")
endif()

# Find GLFW for visualization (required)
find_package(glfw3 REQUIRED)
message(STATUS "Found GLFW3")

# Create executable
add_executable(mujoco_sim_node
    src/mujoco_sim_node.cpp
)

target_include_directories(mujoco_sim_node PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${MUJOCO_INCLUDE_DIR}
)

target_link_libraries(mujoco_sim_node
    ${MUJOCO_LIBRARY}
    glfw
)

ament_target_dependencies(mujoco_sim_node
    rclcpp
    std_msgs
    std_srvs
    sensor_msgs
    geometry_msgs
    nav_msgs
    mujoco_ros_msgs
)

# Install executable
install(TARGETS mujoco_sim_node
    DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY launch
    DESTINATION share/${PROJECT_NAME}
    OPTIONAL
)

# Install config files
install(DIRECTORY config
    DESTINATION share/${PROJECT_NAME}
    OPTIONAL
)

ament_package()
