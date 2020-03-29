cmake_minimum_required(VERSION 3.13)
set(CMAKE_VERBOSE_MAKEFILE On) # enables indexing in eclipse?

project(hourGlass LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
add_compile_options("-Wall")
add_compile_options("-O3")

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(LOCAL_INCLUDE_DIR "include")

#find_package(OpenMP REQUIRED)

file(GLOB SOURCES "src/*.cpp") # new files are read only after modifying this file?
file(GLOB HEADERS "include/*.h")


# TEST:
find_package(Boost REQUIRED unit_test_framework)

# vec3d-test 
file(GLOB SOURCES "test/vec3d-test.cpp" "src/vec3d.cpp") 
add_executable(vec3d-test ${SOURCES})
target_include_directories(vec3d-test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${LOCAL_INCLUDE_DIR} ${Boost_INCLUDE_DIRS})
target_link_libraries(vec3d-test ${Boost_LIBRARIES})

# minimum-test
file(GLOB SOURCES "test/minimum-test.cpp" "src/minimum.cpp")
add_executable(minimum-test ${SOURCES} )
target_include_directories(minimum-test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${Boost_INCLUDE_DIRS} "./include")
target_link_libraries(minimum-test ${Boost_LIBRARIES})

# minimumdistance-test
file(GLOB SOURCES "test/minimumdistance-test.cpp" "src/minimum.cpp" "src/vec3d.cpp")
add_executable(minimumdistance-test ${SOURCES} )
target_include_directories(minimumdistance-test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${Boost_INCLUDE_DIRS} "./include")
target_link_libraries(minimumdistance-test ${Boost_LIBRARIES})
