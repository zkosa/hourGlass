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
file(GLOB MOC_HEADERS "include/mainwindow.h")
set(UI "src/mainwindow.ui") # ui must be placed alongside the cpp file!

include_directories("include")

# TEST:
find_package(Boost REQUIRED unit_test_framework)

# vec3d-test 
file(GLOB SOURCES "test/vec3d-test.cpp" "src/vec3d.cpp") 
add_executable(vec3d-test ${SOURCES})
target_include_directories(vec3d-test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${Boost_INCLUDE_DIRS})

# boundary-test
file(GLOB SOURCES "test/boundary-test.cpp" "src/scene.cpp" "src/cell.cpp" "src/particle.cpp" "src/boundingbox.cpp" "src/vec3d.cpp" "src/boundary_planar.cpp" "src/boundary_axissymmetric.cpp" "src/minimum.cpp" "src/customopenglwidget.cpp" "src/mainwindow.cpp")
add_executable(boundary-test ${SOURCES} ${MOC_HEADERS} ${UI})
target_include_directories(boundary-test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}  ${Boost_INCLUDE_DIRS})
set_target_properties(boundary-test PROPERTIES AUTOMOC TRUE)

# minimum-test
file(GLOB SOURCES "test/minimum-test.cpp" "src/minimum.cpp")
add_executable(minimum-test ${SOURCES} )
target_include_directories(minimum-test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${Boost_INCLUDE_DIRS} "./include")

# minimumdistance-test
file(GLOB SOURCES "test/minimumdistance-test.cpp" "src/minimum.cpp" "src/vec3d.cpp" "src/boundary_axissymmetric.cpp" "src/particle.cpp" "src/vector.cpp" "src/boundingbox.cpp")
add_executable(minimumdistance-test ${SOURCES} )
target_include_directories(minimumdistance-test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${Boost_INCLUDE_DIRS} "./include")

if (UNIX)
   target_link_libraries(vec3d-test ${Boost_LIBRARIES})
   target_link_libraries(boundary-test ${Boost_LIBRARIES} Qt5::Widgets GL)
   target_link_libraries(minimum-test ${Boost_LIBRARIES})
   target_link_libraries(minimumdistance-test ${Boost_LIBRARIES} Qt5::Widgets GL)
elseif (WIN32)
   target_link_libraries(vec3d-test ${Boost_LIBRARIES})
   target_link_libraries(boundary-test ${Boost_LIBRARIES} Qt5::Widgets opengl32)
   target_link_libraries(minimum-test ${Boost_LIBRARIES})
   target_link_libraries(minimumdistance-test ${Boost_LIBRARIES} Qt5::Widgets opengl32)
else ()
	message("Unsupported operating system")
endif ()
