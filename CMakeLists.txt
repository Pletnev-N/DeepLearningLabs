cmake_minimum_required(VERSION 3.5.1)
project(BackPropProject CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
include_directories(include ${OpenCV_INCLUDE_DIRS})

set(SOURCES src/main.cpp)
set(HEADERS include/DigitsDataset.h include/NeuralNet.h)

add_executable(BackProp ${SOURCES} ${HEADERS})
target_link_libraries(BackProp PUBLIC ${OpenCV_LIBS})