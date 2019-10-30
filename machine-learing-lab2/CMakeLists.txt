cmake_minimum_required(VERSION 3.15)
project(machien_learing_lab2)

set(CMAKE_CXX_STANDARD 17)

include_directories(./eigen)
set(SOURCE_FILES main.cpp Data.cpp Data.h ProductStrategy.cpp ProductStrategy.h BasicSettings.h LearingAlgorithm.h LogisticRegression.cpp LogisticRegression.h MultivariateNormalRandomVariable.h LearingAlgorithm.cpp)
add_executable(machine_learing_lab2 ${SOURCE_FILES} )