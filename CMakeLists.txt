cmake_minimum_required(VERSION 3.13)
set(CMAKE_VERBOSE_MAKEFILE On) # enables indexing in eclipse?

project(hourGlass LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
add_compile_options("-Wall")
add_compile_options("-O3")
#add_compile_options("-Wpedantic" "-Wextra" "-fexceptions")
#string(APPEND CMAKE_CXX_FLAGS "-Wall")
#string(APPEND CMAKE_CXX_FLAGS_DEBUG " -g3")
#string(APPEND CMAKE_CXX_FLAGS_RELEASE " -O3")

if (WIN32)
	set(CMAKE_PREFIX_PATH "C:\\Qt\\Qt5.13.2\\5.13.2\\msvc2017_64") # add also to system path (Win)
endif (WIN32)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt5 REQUIRED Core Gui Widgets)
#find_package(OpenMP REQUIRED)

file(GLOB SOURCES "src/*.cpp") # new files are read only after modifying this file?
file(GLOB HEADERS "include/*.h")
set(UI "src/mainwindow.ui") # ui must be placed alongside the cpp file!

include_directories("include")

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS} ${UI}) # are the headers needed?
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${OPENGL_INCLUDE_DIR})
set_target_properties(${PROJECT_NAME} PROPERTIES AUTOMOC TRUE)

if (UNIX)
    target_link_libraries(${PROJECT_NAME} Qt5::Core Qt5::Gui Qt5::Widgets GL) # OpenMP::OpenMP_CXX
elseif (WIN32)
    target_link_libraries(${PROJECT_NAME} Qt5::Core Qt5::Gui Qt5::Widgets opengl32) # OpenMP::OpenMP_CXX
else ()
	message("Unsupported operating system")
endif ()
