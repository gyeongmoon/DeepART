cmake_minimum_required (VERSION 2.8)

project (DeepART)

include_directories(./include)

add_executable(./deep_art ./src/main.cpp)

add_definitions(-Wall -Wextra -ofast -std=c++11)
