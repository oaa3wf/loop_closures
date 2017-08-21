cmake_minimum_required(VERSION 2.8)
project(simple_loops)

add_definitions(-std=c++11 -Wall)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")
#set(OpenCV_DIR /home/sastrygrp/code/3rd_party/opencv-3.1.0/release)
find_package(OpenCV 3 REQUIRED)

set(DEPENDENCY_DIR ${CMAKE_CURRENT_BINARY_DIR}/dependencies)
set(DEPENDENCY_INSTALL_DIR ${DEPENDENCY_DIR}/install)

#if(FALSE)
#	macro(GetDependency name other_dependency)
#	  find_package(${name} QUIET 
#		PATHS ${DEPENDENCY_INSTALL_DIR} /home/sastrygrp/code/3rd_party/DBoW2 /home/sastrygrp/code/3rd_party/DBoW2/dependencies/src/DLib )
#	  if(${${name}_FOUND})
#		message("${name} library found, using it from the system")
#		include_directories(${${name}_INCLUDE_DIRS})
#		add_custom_target(${name})
#	  else(${${name}_FOUND})
#		message("${name} library not found in the system, it will be downloaded on build")
#		option(DOWNLOAD_${name}_dependency "Download ${name} dependency" OFF)
#		if(${DOWNLOAD_${name}_dependency})
#		  ExternalProject_Add(${name}
#		    PREFIX ${DEPENDENCY_DIR}
#		    GIT_REPOSITORY http://github.com/dorian3d/${name}
#		    GIT_TAG v1.1-nonfree
#		    INSTALL_DIR ${DEPENDENCY_INSTALL_DIR}
#		    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
#		    DEPENDS ${other_dependency})
#		else()
#		  message(SEND_ERROR "Please, activate DOWNLOAD_${name}_dependency option or download manually")
#		endif(${DOWNLOAD_${name}_dependency})
#	  endif(${${name}_FOUND})
#	endmacro(GetDependency)
#endif(FALSE)


#GetDependency(DLib "")
#GetDependency(DBoW2 DLib)


#find_package(DLib REQUIRED)
#if(${DLib_FOUND})
#		message("DLib library found, using it from the system")
#		include_directories(${DLib_INCLUDE_DIRS})
		#add_custom_target(${DBoW2})
#endif(${DLib_FOUND})

#set(DBoW2_DIR /home/sastrygrp/code/3rd_party/DBoW2)
#find_package(DBoW2 REQUIRED)
#if(${DBoW2_FOUND})
#		message("DBoW2 library found, using it from the system")
#		include_directories(${DBoW2_INCLUDE_DIRS})
#		#add_custom_target(${DBoW2})
#endif(${DBoW2_FOUND})

#add_custom_target(Dependencies
#  ${CMAKE_COMMAND} ${CMAKE_SOURCE_DIR}
#DEPENDS DBoW2 DLib)

#include_directories(
#${DBoW2_DIR}/install/include
#${DLib_DIR}/../DLib/install/include)

find_package(CUDA QUIET REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})


find_package(Caffe REQUIRED)
include_directories(${Caffe_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS}) 

find_package(Flann REQUIRED)
include_directories(${Flann_INCLUDE_DIRS})


get_target_property(CAFFE_SONAME caffe IMPORTED_SONAME_RELEASE)
message( STATUS "CAFFE_SONAME is: " ${soname})

get_target_property(version caffe SOVERSION)
message( STATUS "VERSION is: " ${version})

if("${CAFFE_SONAME}" MATCHES "libcaffe.so.([0-9]+).([0-9]+).([0-9]+)(-(.+))?")
    set(CAFFE_MAJOR "${CMAKE_MATCH_1}")
    set(CAFFE_MINOR "${CMAKE_MATCH_2}")
    set(CAFFE_PATCH "${CMAKE_MATCH_3}")
    set(CAFFE_PATCH2 "${CMAKE_MATCH_5}")
    set(CAFFE_VERSION_STRING "${CAFFE_MAJOR}.${CAFFE_MINOR}.${CAFFE_PATCH}")
    if(NOT "${CAFFE_PATCH2}" STREQUAL "")
        set(CAFFE_VERSION_STRING "${CAFFE_VERSION_STRING}-${CAFFE_PATCH2}")
    endif()
else()
    set(CAFFE_MAJOR 0)
    set(CAFFE_MINOR 0)
    set(CAFFE_PATCH 0)
    set(CAFFE_PATCH2 0)
    set(CAFFE_VERSION_STRING unknown)
endif()

math(EXPR CAFFE_VERSION "${CAFFE_MAJOR}*10000 + ${CAFFE_MINOR}*100 + ${CAFFE_PATCH}")
message(${CAFFE_PATCH})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/caffe-version.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/caffe-version.h
               @ONLY)
set(CMAKE_INCLUDE_CURRENT_DIR TRUE)

#add_executable(simple_loops
#    src/simple_loops.cpp)
#target_link_libraries(simple_loops
#   ${OpenCV_LIBS}
#    ${DLib_LIBS}
#	${DBoW2_LIBS})

add_executable(forward_propagate src/forward_propagate.cpp )
target_link_libraries(forward_propagate
    ${OpenCV_LIBS}
	${Caffe_LIBRARIES})

add_executable(test_deepFeatures src/test_deepFeatures.cpp  src/DeepFeatures.cpp)
target_link_libraries(test_deepFeatures
    ${OpenCV_LIBS}
    ${Caffe_LIBRARIES})

add_executable(findNearestNeighbors src/findNearestNeighbors.cpp  src/DeepFeatures.cpp)
target_link_libraries(findNearestNeighbors
    ${OpenCV_LIBS}
    ${Caffe_LIBRARIES}
		${Flann_LIBRARIES})
