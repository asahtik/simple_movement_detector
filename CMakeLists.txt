cmake_minimum_required(VERSION 3.0.0)
project(movement_detector VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)

include(CTest)
enable_testing()

find_package(OpenCV REQUIRED)

message(STATUS "OpenCV library status:")
message(STATUS "    dirs:  ${OpenCV_INCLUDE_DIRS}")

file(GLOB HEADERIMPLS "src/*.hpp")

message(STATUS "Header files:")
message(STATUS ${HEADERIMPLS})

add_executable(movement_detector src/main.cpp ${HEADERIMPLS})
target_include_directories(movement_detector PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(movement_detector ${OpenCV_LIBS})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
