cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(test_features)

find_package(PCL 1.2 REQUIRED)

FIND_PACKAGE(Boost COMPONENTS program_options REQUIRED)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS};${PCL_INCLUDE_DIRS})
#INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (test_features test_features.cpp)
TARGET_LINK_LIBRARIES(test_features ${Boost_LIBRARIES};${PCL_LIBRARIES})
#TARGET_LINK_LIBRARIES(${Boost_LIBRARIES})

#include_directories(${PCL_INCLUDE_DIRS})
#link_directories(${PCL_LIBRARY_DIRS})
#add_definitions(${PCL_DEFINITIONS})

#add_executable (test_features test_features.cpp)
#target_link_libraries (test_features ${PCL_LIBRARIES})	
