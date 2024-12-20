cmake_minimum_required(VERSION 3.15)

project(acts)

set(SOURCES
  acts.cpp)

set(HEADERS
  ../../cnf.hpp
  acts.hpp)

add_library(acts SHARED ${SOURCES} ${HEADERS})

set_target_properties(${PROJECT_NAME} PROPERTIES
                     CXX_STANDARD 17
                     CXX_STANDARD_REQUIRED ON)