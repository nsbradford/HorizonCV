cmake_minimum_required(VERSION 2.8)
project( horcpp )

if(UNIX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=gnu++0x")
endif()

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( horcpp horizon.cpp )
target_link_libraries( horcpp ${OpenCV_LIBS} )