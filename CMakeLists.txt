cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(mass_spring_explicit)

add_definitions(-DUNICODE -D_UNICODE)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# CUDA
find_package(CUDA REQUIRED)

#OpenGL
find_package(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIR})
include_directories("D:/OpenGL/include")
link_directories("D:/OpenGL/lib")
link_libraries(glew32 glfw3 soil2 ${OPENGL_LIBRARY})

# Eigen
#INCLUDE_DIRECTORIES("D:/eigen-3.3.8/install/include")
#include_directories(/usr/local/include)


AUX_SOURCE_DIRECTORY(. SRC_LIST)
CUDA_ADD_EXECUTABLE(mass_spring_explicit ${SRC_LIST})
#CUDA_ADD_EXECUTABLE
#ADD_EXECUTABLE