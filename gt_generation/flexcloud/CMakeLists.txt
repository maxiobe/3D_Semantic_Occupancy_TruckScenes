cmake_minimum_required(VERSION 3.8)
project(flexcloud)

# Default to C++20
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()
set(CMAKE_BUILD_TYPE Release)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-local-typedefs)
endif()

# find dependencies
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(CGAL REQUIRED)
find_package(Boost REQUIRED)
find_package(GeographicLib REQUIRED)
find_package(pybind11 REQUIRED)

# Fetch rerun_sdk and yaml-cpp
include(FetchContent)
set(BUILD_SHARED_LIBS ON CACHE BOOL "Build rerun_sdk as a shared library" FORCE)
FetchContent_Declare(rerun_sdk URL https://github.com/rerun-io/rerun/releases/latest/download/rerun_cpp_sdk.zip)
FetchContent_Declare(
  yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG 0.8.0
)
FetchContent_MakeAvailable(rerun_sdk yaml-cpp)

include_directories(SYSTEM
  ${PCL_INCLUDE_DIRS}
  ${GeographicLib_INCLUDE_DIRS}
)

####################################
# file_io
####################################

add_library(file_io SHARED
  src/file_io.cpp
)

target_include_directories(file_io
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/flexcloud>
    $<INSTALL_INTERFACE:include>)

target_link_libraries(file_io
  Eigen3::Eigen
  ${PCL_LIBRARIES}
  ${GeographicLib_LIBRARIES}
  ${Boost_LIBRARIES})

####################################
# map_transformation
####################################
add_library(transform SHARED
  src/transform.cpp
)

target_include_directories(transform
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/flexcloud>
    $<INSTALL_INTERFACE:include>)

target_link_libraries(transform
  Eigen3::Eigen
  ${PCL_LIBRARIES}
  CGAL::CGAL)

####################################
# visualization
####################################

add_library(visualization SHARED
  src/visualization.cpp
)

target_include_directories(visualization
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/flexcloud>
    $<INSTALL_INTERFACE:include>)

target_link_libraries(visualization
  Eigen3::Eigen
  ${PCL_LIBRARIES}
  CGAL::CGAL
  ${Boost_LIBRARIES}
  rerun_sdk)

####################################
# keyframe_interpolation
####################################

add_library(keyframe_interpolation SHARED
src/keyframe_interpolation.cpp
)

target_include_directories(keyframe_interpolation
PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/flexcloud>
  $<INSTALL_INTERFACE:include>)

target_link_libraries(keyframe_interpolation
  file_io
  visualization
  Eigen3::Eigen
  yaml-cpp::yaml-cpp
  ${PCL_LIBRARIES})

####################################
# analysis
####################################

add_library(analysis SHARED
  src/analysis.cpp
)

target_include_directories(analysis
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/flexcloud>
    $<INSTALL_INTERFACE:include>)

target_link_libraries(analysis
  Eigen3::Eigen)


######################################################################
# NEW SECTION: Python Bindings
######################################################################

# 1. Add the new FlexCloud wrapper library
add_library(flexcloud_lib SHARED
    src/flexcloud.cpp
)
target_include_directories(flexcloud_lib
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> # Include top-level for "flexcloud/flexcloud.hpp"
    $<INSTALL_INTERFACE:include>
)
# Link flexcloud_lib against the other components it uses
target_link_libraries(flexcloud_lib
    keyframe_interpolation
    transform
    file_io
    visualization
    analysis
    pcd_georef_lib
)


# pcd_georef_lib (NEW LIBRARY)
# This library contains the core logic from pcd_georef.cpp
add_library(pcd_georef_lib SHARED
  src/pcd_georef.cpp
)
target_include_directories(pcd_georef_lib
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/flexcloud>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(pcd_georef_lib
  yaml-cpp::yaml-cpp
  file_io
  transform
  visualization
  analysis)

# 2. Create the Python module using pybind11
pybind11_add_module(flexcloud_bindings SHARED src/bindings.cpp)

# 3. Link the Python module to your wrapper library
target_link_libraries(flexcloud_bindings PRIVATE flexcloud_lib)

install(TARGETS flexcloud_bindings
        LIBRARY DESTINATION "lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages")

######################################################################

install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY include/
  DESTINATION include
)

install(TARGETS
  file_io
  keyframe_interpolation
  transform
  visualization
  analysis
  flexcloud_lib # <--- ADD flexcloud_lib TO INSTALL
  pcd_georef_lib
  EXPORT export_${PROJECT_NAME}
  DESTINATION lib/${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

####################################
# Main Executables
####################################

# select_keyframes
add_executable(select_keyframes
  src/select_keyframes.cpp
)

target_include_directories(select_keyframes
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/flexcloud>
    $<INSTALL_INTERFACE:include>)

target_link_libraries(select_keyframes
  file_io
  keyframe_interpolation
  visualization)

# pcd_georef (EXECUTABLE)
# This now just links against the library which contains its main() function
add_executable(pcd_georef
  src/pcd_georef.cpp
)
target_link_libraries(pcd_georef pcd_georef_lib)

install(TARGETS
  pcd_georef
  select_keyframes
  DESTINATION pcd_georef/${PROJECT_NAME})