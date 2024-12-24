cmake_minimum_required(VERSION 3.15)

project(net)

add_subdirectory(acts)
add_subdirectory(loss)

set(SOURCES
  net.cpp)

set(HEADERS
  net.hpp)

add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})

set_target_properties(${PROJECT_NAME} PROPERTIES
                     CXX_STANDARD 17
                     CXX_STANDARD_REQUIRED ON)

target_link_libraries (${PROJECT_NAME} PUBLIC 
                       acts
                       loss)
