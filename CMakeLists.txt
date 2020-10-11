cmake_minimum_required(VERSION 3.13)
set(CMAKE_VERBOSE_MAKEFILE On) # enables indexing in eclipse?

project(hourGlass LANGUAGES CXX CUDA)

set(TESTS ON) # ON / OFF
set(BENCHMARKS ON) # ON / OFF
set(CODE_COVERAGE_ANALYSIS OFF) # ON / OFF, linux only

set(CMAKE_CXX_STANDARD 20)
# add_compile_options("-Wall") # it would be passed to nvcc too!
# disabling optimization in order to improve the fidelity of coverage analysis:
if (NOT CODE_COVERAGE_ANALYSIS)
	#add_compile_options("-O3")
endif(NOT CODE_COVERAGE_ANALYSIS)
#add_compile_options("-Wpedantic" "-Wextra" "-fexceptions")
string(APPEND CMAKE_CXX_FLAGS "-Wall")
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
find_package(OpenGL REQUIRED)

message(STATUS "OPENGL_INCLUDE_DIR:= ${OPENGL_INCLUDE_DIR}")  # OPENGL_INCLUDE_DIR is not set on Windows, except for cygwin!
message(STATUS "OPENGL_LIBRARY:= ${OPENGL_LIBRARY}")
message(STATUS "OPENGL_LIBRARIES:= ${OPENGL_LIBRARIES}")


file(GLOB SOURCES "src/*.cpp")
file(GLOB CUDA_SOURCES "src/*.cu")
file(GLOB HEADERS "include/*.h")
file(GLOB MOC_HEADERS "include/mainwindow.h")
set(UI "src/mainwindow.ui") # ui must be placed alongside the cpp file!

include_directories("include")

# workaround because of cmake/Qt bug before Qt 5.14.1 and 5.15 Alpha
# https://gitlab.kitware.com/cmake/cmake/-/issues/16915
if ( TARGET Qt5::Core )
	get_property( core_options TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS )
	string( REPLACE "-fPIC" "" new_core_options "${core_options}" )
	set_property( TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS ${new_core_options} )
	set_property( TARGET Qt5::Core PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE "ON" )
	set( CMAKE_CXX_COMPILE_OPTIONS_PIE "-fPIC" )
endif()


add_executable(${PROJECT_NAME} ${SOURCES} ${CUDA_SOURCES} ${MOC_HEADERS} ${UI}) # the MOC headers are needed
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${OPENGL_INCLUDE_DIR})
set_target_properties(${PROJECT_NAME} PROPERTIES AUTOMOC TRUE)
target_link_libraries(${PROJECT_NAME} Qt5::Core Qt5::Gui Qt5::Widgets ${OPENGL_LIBRARIES}) # OpenMP::OpenMP_CXX


##############################
# TESTING: ###################
##############################
if(TESTS)

find_package(Boost REQUIRED unit_test_framework)
enable_testing()

# macro adds tests and collects the targets to a variable
macro(add_unit_test UNIT_TEST)
    set(UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} ${UNIT_TEST} )   # PARENT_SCOPE removed because of no subfolderss
    add_test(${UNIT_TEST} ${CMAKE_CURRENT_BINARY_DIR}/${UNIT_TEST})
endmacro(add_unit_test)

# vec3d-test 
file(GLOB SOURCES "test/vec3d-test.cpp" "src/vec3d.cpp" "src/vecaxisym.cpp") 
add_executable(vec3d-test ${SOURCES})
target_include_directories(vec3d-test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${Boost_INCLUDE_DIRS})
target_link_libraries(vec3d-test ${Boost_LIBRARIES})
add_unit_test(vec3d-test)

# particle-test
file(GLOB SOURCES "test/particle-test.cpp" "src/particle.cpp" "src/scene.cpp" "src/cell.cpp" "src/boundingbox.cpp" "src/vec3d.cpp" "src/boundary_planar.cpp" "src/boundary_axissymmetric.cpp" "src/minimum.cpp" "src/mainwindow.cpp" "src/customopenglwidget.cpp" "src/cuda.cu")
add_executable(particle-test ${SOURCES} ${MOC_HEADERS} ${UI})
target_include_directories(particle-test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${Boost_INCLUDE_DIRS})
set_target_properties(particle-test PROPERTIES AUTOMOC TRUE)
target_link_libraries(particle-test ${Boost_LIBRARIES} Qt5::Widgets ${OPENGL_LIBRARIES})
add_unit_test(particle-test)

# cell-test
file(GLOB SOURCES "test/cell-test.cpp" "src/cell.cpp" "src/particle.cpp" "src/boundingbox.cpp" "src/vec3d.cpp" "src/vecaxisym.cpp" "src/boundary_planar.cpp" "src/boundary_axissymmetric.cpp" "src/minimum.cpp" "src/cuda.cu")
add_executable(cell-test ${SOURCES})
target_include_directories(cell-test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${Boost_INCLUDE_DIRS})
target_link_libraries(cell-test ${Boost_LIBRARIES} Qt5::Widgets ${OPENGL_LIBRARIES})
add_unit_test(cell-test)

# scene-test
file(GLOB SOURCES "test/scene-test.cpp" "src/scene.cpp" "src/cell.cpp" "src/particle.cpp" "src/boundingbox.cpp" "src/vec3d.cpp" "src/boundary_planar.cpp" "src/boundary_axissymmetric.cpp" "src/minimum.cpp" "src/mainwindow.cpp" "src/customopenglwidget.cpp" "src/cuda.cu")
add_executable(scene-test ${SOURCES} ${MOC_HEADERS} ${UI})
target_include_directories(scene-test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${Boost_INCLUDE_DIRS})
set_target_properties(scene-test PROPERTIES AUTOMOC TRUE)
target_link_libraries(scene-test ${Boost_LIBRARIES} Qt5::Widgets ${OPENGL_LIBRARIES})
add_unit_test(scene-test)

# boundary-test
file(GLOB SOURCES "test/boundary-test.cpp" "src/scene.cpp" "src/cell.cpp" "src/particle.cpp" "src/boundingbox.cpp" "src/vec3d.cpp" "src/boundary_planar.cpp" "src/boundary_axissymmetric.cpp" "src/minimum.cpp" "src/customopenglwidget.cpp" "src/mainwindow.cpp" "src/cuda.cu")
add_executable(boundary-test ${SOURCES} ${MOC_HEADERS} ${UI})
target_include_directories(boundary-test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}  ${Boost_INCLUDE_DIRS})
set_target_properties(boundary-test PROPERTIES AUTOMOC TRUE)
target_link_libraries(boundary-test ${Boost_LIBRARIES} Qt5::Widgets ${OPENGL_LIBRARIES})
add_unit_test(boundary-test)

# minimum-test
file(GLOB SOURCES "test/minimum-test.cpp" "src/minimum.cpp")
add_executable(minimum-test ${SOURCES} )
target_include_directories(minimum-test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${Boost_INCLUDE_DIRS})
target_link_libraries(minimum-test ${Boost_LIBRARIES})
add_unit_test(minimum-test)

# minimumdistance-test
file(GLOB SOURCES "test/minimumdistance-test.cpp" "src/minimum.cpp" "src/vec3d.cpp" "src/boundary_planar.cpp" "src/boundary_axissymmetric.cpp" "src/particle.cpp" "src/vector.cpp" "src/boundingbox.cpp")
add_executable(minimumdistance-test ${SOURCES} )
target_include_directories(minimumdistance-test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${Boost_INCLUDE_DIRS})
target_link_libraries(minimumdistance-test ${Boost_LIBRARIES} Qt5::Widgets ${OPENGL_LIBRARIES})
add_unit_test(minimumdistance-test)

# cuda-test
add_executable(cuda-test ${CUDA_SOURCES} ${MOC_HEADERS} ${UI} "test/cuda-test.cpp" "src/scene.cpp" "src/cell.cpp" "src/particle.cpp" "src/boundingbox.cpp" "src/vec3d.cpp" "src/boundary_planar.cpp" "src/boundary_axissymmetric.cpp" "src/minimum.cpp" "src/mainwindow.cpp" "src/customopenglwidget.cpp")
target_include_directories(cuda-test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${Boost_INCLUDE_DIRS})
set_target_properties(cuda-test PROPERTIES AUTOMOC TRUE)
target_link_libraries(cuda-test ${Boost_LIBRARIES} Qt5::Widgets ${OPENGL_LIBRARIES})
target_compile_options(cuda-test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
#                       --use_fast_math
#                       --relocatable-device-code=true
                       >)
add_unit_test(cuda-test)

# passing-test
file(GLOB SOURCES "test/passing-test.cpp")
add_executable(passing-test ${SOURCES} )
target_include_directories(passing-test PUBLIC ${Boost_INCLUDE_DIRS})
target_link_libraries(passing-test ${Boost_LIBRARIES})
add_unit_test(passing-test)

# failing-test
set(ADD_FAILING_TEST OFF)
if(ADD_FAILING_TEST)
	file(GLOB SOURCES "test/failing-test.cpp")
	add_executable(failing-test ${SOURCES} )
	target_include_directories(failing-test PUBLIC ${Boost_INCLUDE_DIRS})
	target_link_libraries(failing-test ${Boost_LIBRARIES})
	add_unit_test(failing-test)
endif(ADD_FAILING_TEST)

# boosttest-test
set(TEST_THE_TEST OFF)
if(TEST_THE_TEST)
	file(GLOB SOURCES "test/boosttest-test.cpp")
	add_executable(boosttest-test ${SOURCES} )
	target_include_directories(boosttest-test PUBLIC ${Boost_INCLUDE_DIRS})
	target_link_libraries(boosttest-test ${Boost_LIBRARIES})
	add_unit_test(boosttest-test)
endif(TEST_THE_TEST)

# object_size
file(GLOB SOURCES "test/object_size.cpp" "src/vec3d.cpp" "src/particle.cpp" "src/boundingbox.cpp" "src/boundary_planar.cpp" "src/boundary_axissymmetric.cpp" "src/minimum.cpp" "src/cell.cpp" "src/scene.cpp" "src/mainwindow.cpp" "src/customopenglwidget.cpp" "src/cuda.cu") #  "src/boundingbox.cpp" "src/boundary_planar.cpp" "src/boundary_axissymmetric.cpp" "src/minimum.cpp"
add_executable(object_size ${SOURCES} ${MOC_HEADERS} ${UI})
target_include_directories(object_size PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(object_size ${Boost_LIBRARIES} Qt5::Widgets ${OPENGL_LIBRARIES})

############
# running tests automatically after build:
set(UNIT_TEST_COLLECTOR all_tests)
add_custom_target( ${UNIT_TEST_COLLECTOR} ALL DEPENDS ${UNIT_TEST_TARGETS})
add_custom_command(
      TARGET ${UNIT_TEST_COLLECTOR}
      COMMENT "Run tests"
      POST_BUILD 
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      COMMAND ${CMAKE_CTEST_COMMAND} ARGS --output-on-failure
)

if(CMAKE_COMPILER_IS_GNUCXX AND CODE_COVERAGE_ANALYSIS)
	# creating coverage analysis target (for the test):
	set(COVERAGE_COMPILER_FLAGS "-g -fprofile-arcs -ftest-coverage"
    	CACHE INTERNAL "")
    set(COVERAGE_TARGET_NAME "coverage")
	include(CodeCoverage.cmake)
	APPEND_COVERAGE_COMPILER_FLAGS()
	SETUP_TARGET_FOR_COVERAGE_LCOV(
	        NAME ${COVERAGE_TARGET_NAME}               
	        EXECUTABLE ${CMAKE_CTEST_COMMAND}
	        DEPENDENCIES ${UNIT_TEST_COLLECTOR}
	        )
	        
# run the coverage analysis target manually after build:
# run "make coverage" from bin
# or build the coverage target from eclipse

endif(CMAKE_COMPILER_IS_GNUCXX AND CODE_COVERAGE_ANALYSIS)
endif(TESTS)

if(BENCHMARKS)
	find_package(benchmark REQUIRED)
	file(GLOB SOURCES "test/benchmark.cpp" "src/*.cpp")
	list(FILTER SOURCES EXCLUDE REGEX "src/main.cpp") # exclude main
	add_executable(${PROJECT_NAME}_benchmark ${SOURCES} ${MOC_HEADERS} ${UI}) # the MOC headers are needed
	target_include_directories(${PROJECT_NAME}_benchmark PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${OPENGL_INCLUDE_DIR})
	set_target_properties(${PROJECT_NAME}_benchmark PROPERTIES AUTOMOC TRUE)
	target_link_libraries(${PROJECT_NAME}_benchmark benchmark::benchmark Qt5::Core Qt5::Gui Qt5::Widgets ${OPENGL_LIBRARIES})
endif(BENCHMARKS)

# run the benchmark with constant CPU frequency:
# sudo cpupower frequency-set --governor performance
# ./bin/bin/hourGlass_benchmark
# sudo cpupower frequency-set --governor powersave
