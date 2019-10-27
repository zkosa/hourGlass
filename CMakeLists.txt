cmake_minimum_required(VERSION 3.1)
project(hourGlass LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3")
#set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")


set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

add_subdirectory(Dependencies/glfw)

find_package(OpenGL REQUIRED)

find_package(Qt5 COMPONENTS Core Quick REQUIRED)

 
#Bring the headers into the project
include_directories(include Dependencies/glfw/include)
 
#Can manually add the sources using the set command as follows:
#set(SOURCES src/mainapp.cpp src/Student.cpp)
 
#However, the file(GLOB...) allows for wildcard additions:
file(GLOB SOURCES "src/*.cpp")
 
add_executable(hourGlass ${SOURCES})

#(GL_LIB "glfw")
#target_link_libraries(hourGlass "${GL_LIB}")
target_link_libraries(hourGlass glfw)
target_include_directories(hourGlass PUBLIC ${OPENGL_INCLUDE_DIR})
target_link_libraries(hourGlass ${OPENGL_gl_LIBRARY})

