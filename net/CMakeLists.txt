cmake_minimum_required(VERSION 3.15)

project(net)

add_subdirectory(acts)
add_subdirectory(loss)
add_subdirectory(layers)

set(SOURCES
  net.cpp)

set(HEADERS
  net.hpp)

add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})

find_package(nlohmann_json 3.2.0 REQUIRED)

set_target_properties(${PROJECT_NAME} PROPERTIES
                     CXX_STANDARD 17
                     CXX_STANDARD_REQUIRED ON)

target_link_libraries (${PROJECT_NAME} PUBLIC 
                       acts
                       loss
                       layers
                       nlohmann_json::nlohmann_json)
