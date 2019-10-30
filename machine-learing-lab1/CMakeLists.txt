cmake_minimum_required(VERSION 3.14)
project(machine_learing_lab1)

set(CMAKE_CXX_STANDARD 14)

include_directories(./eigen)
set(SOURCE_FILES main.cpp Data.cpp Data.h ProductStrategy.cpp ProductStrategy.h BasicSettings.h LeastSquare.cpp LeastSquare.h)
add_executable(Clion ${SOURCE_FILES} Data.cpp Data.h)