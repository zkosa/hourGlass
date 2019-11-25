cmake_minimum_required(VERSION 3.13)
set(CMAKE_VERBOSE_MAKEFILE On) # enabling indexing in eclipse?

project(hourGlass LANGUAGES CXX)
#set(PROJECT_NAME test-qt-widgets)
#project(${PROJECT_NAME})

#set(GUI gui)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -O3")

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# Instruct CMake to run moc automatically when needed.
set(CMAKE_AUTOMOC ON)
# Instruct CMake to run uic automatically when needed.
set(CMAKE_AUTOUIC ON)

#set_target_properties(${GUI} PROPERTIES AUTOMOC TRUE)

find_package(Qt5 REQUIRED Core Gui Widgets)
#find_package(OpenMP REQUIRED)


file(GLOB SOURCES "src/*.cpp") # new files are read only after modifying this file?
file(GLOB HEADERS "include/*.h")
set(UI "src/mainwindow.ui") # ui must be placed alongside the cpp file

include_directories("include")


#add_library(${GUI}  SHARED  mainwindow.h  mainwindow.cpp)

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS} ${UI}) # are the headers needed?
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} /usr/include/x86_64-linux-gnu/qt5/  ${OPENGL_INCLUDE_DIR}) # headers
#set_target_properties(${PROJECT_NAME} PROPERTIES  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set_target_properties(${PROJECT_NAME} PROPERTIES AUTOMOC TRUE)

#qt5_use_modules(${PROJECT_NAME} Core Gui Widgets)
target_link_libraries( ${PROJECT_NAME} Qt5::Core Qt5::Gui Qt5::Widgets ${Qt5Core_QTMAIN_LIBRARIES} GL) # OpenMP::OpenMP_CXX) # ${OPENGL_gl_LIBRARY}PRIVATE 
#message(STATUS "OPENGL_gl_LIBRARY=${OPENGL_gl_LIBRARY}")

#[[
get_cmake_property(_variableNames VARIABLES)
list (SORT _variableNames)
foreach (_variableName ${_variableNames})
    message(STATUS "${_variableName}=${${_variableName}}")
endforeach()
]]
message(STATUS "SOURCES=${SOURCES}")
message(STATUS "HEADERS=${HEADERS}")
message(STATUS "UI=${UI}")

set(_variableNames Qt5Widgets_VERSION Qt5Widgets_LIBRARIES Qt5Widgets_INCLUDE_DIRS Qt5Widgets_DEFINITIONS Qt5Widgets_COMPILE_DEFINITIONS Qt5Widgets_FOUND Qt5Widgets_EXECUTABLE_COMPILE_FLAGS QT_VISIBILITY_AVAILABLE QT_LIBINFIX)
list (SORT _variableNames)
foreach (_variableName ${_variableNames})
    message(STATUS "${_variableName}=${${_variableName}}")
endforeach()
message(STATUS "SOURCES=${SOURCES}")
