cmake_minimum_required(VERSION 3.15)

project(loss)

set(SOURCES
  loss.cpp)

set(HEADERS
  ../../cnf.hpp
  loss.hpp)

add_library(loss SHARED ${SOURCES} ${HEADERS})

set_target_properties(${PROJECT_NAME} PROPERTIES
                     CXX_STANDARD 17
                     CXX_STANDARD_REQUIRED ON)