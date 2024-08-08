cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

add_definitions(-std=c++11)

# guard against in-source builds
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message(FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there. You may need to remove CMakeCache.txt. ")
endif()

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")
set(OpenCV_DIR "C:/OpenCV-4.6.0/opencv-4.6.0/build")

project(camera_fusion)

find_package(PCL 1.7 REQUIRED)
find_package(Boost 1.58 REQUIRED)
find_package(OpenCV 4.1 REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development NumPy)

#find_package(Matplot++ REQUIRED)

####### matplotplusplus #######  // in-case find_package does not work
#include(FetchContent)

#FetchContent_Declare(matplotplusplus
#       GIT_REPOSITORY https://github.com/alandefreitas/matplotplusplus
#       GIT_TAG origin/master) # or whatever tag you want

#FetchContent_GetProperties(matplotplusplus)
#if(NOT matplotplusplus_POPULATED)
#    FetchContent_Populate(matplotplusplus)
#    add_subdirectory(${matplotplusplus_SOURCE_DIR} ${matplotplusplus_BINARY_DIR} EXCLUDE_FROM_ALL)
#endif()

include_directories(${OpenCV_INCLUDE_DIRS}
				${Boost_INCLUDE_DIRS}
				${PCL_INCLUDE_DIRS})
                
include_directories(${Python3_INCLUDE_DIRS} include)
				
link_directories(${OpenCV_LIBRARY_DIRS}
				 ${PCL_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS}
			    ${PCL_DEFINITIONS})
list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")

# Executable for create matrix exercise
add_executable (3D_object_tracking src/camFusion_Student.cpp src/FinalProject_Camera.cpp src/lidarData.cpp src/matching2D_Student.cpp src/objectDetection2D.cpp)
target_link_libraries (3D_object_tracking PUBLIC
											${OpenCV_LIBRARIES}
											${PCL_LIBRARIES}
											${Python3_LIBRARIES})