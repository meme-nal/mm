cmake_minimum_required(VERSION 3.15)

project(layers)

set(SOURCES
  dense.cpp)

set(HEADERS
  ../../cnf.hpp
  ../../lin.hpp
  layer.hpp
  dense.hpp)

add_library(layers SHARED ${SOURCES} ${HEADERS})

find_package(nlohmann_json 3.2.0 REQUIRED)

set_target_properties(${PROJECT_NAME} PROPERTIES
                     CXX_STANDARD 17
                     CXX_STANDARD_REQUIRED ON)

target_link_libraries (${PROJECT_NAME} PUBLIC 
                       nlohmann_json::nlohmann_json)