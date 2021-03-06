# CMake configuration for the ShapeRegistration project
cmake_minimum_required(VERSION 2.8)

project(ShapeRegistration)

option(GPU "compile for lightning GPU speeds" ON)

# Include the lmfit from the 3rdparty directory (used for the Levenberg-Marquardt solver)
# It needs to be configured and built from source.
include(ExternalProject)
ExternalProject_Add(
	lmfit-6.1
	SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/lmfit-6.1
	CONFIGURE_COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/lmfit-6.1/configure --prefix=${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/lmfit-6.1
	PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/lmfit-6.1
	BUILD_COMMAND make
	BUILD_IN_SOURCE 1
)

# Include the dependencies of lmfit
include_directories(3rdparty/lmfit-6.1/include)
file(GLOB lm 3rdparty/lmfit-6.1/lib/*.so)

# Check if OpenCV is available (used for image I/O)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# Alternatively, check if libpng is available (not yet implemented)
#find_package(libpng REQUIRED)

set(STANDARD_FLAGS ${STANDARD_FLAGS} --std=c++11)

# Check if CUDA is available
if(GPU)
	message( "creating Makefile for lightning speed GPU implementation")
	find_package(CUDA REQUIRED)
	if(CUDA_FOUND)

		# Set the compilation flags for CUDA
		set ( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${STANDARD_FLAGS} --ptxas-options=-v --use_fast_math
			--compiler-options -Wall )

		# Specify the binary name and the source files to build
		cuda_add_executable(sr src/main.cu src/helper.cu src/helper.h src/shapeRegistration.cu src/shapeRegistration.h src/shapeRegistrationGPU.h src/shapeRegistrationGPU.cu) 
		target_link_libraries(sr ${lm}  ${CUDA_LIBARIES} ${OpenCV_LIBS})

	else()
		# CUDA not found
		message(CRITICAL_ERROR "CUDA cannot be found, please ensure CUDA dependencies are installed or compile with CMake option GPU = OFF. CMake will exit now.")

	endif()
else()
	message( "creating Makefile for turtle speed CPU implementation")
endif()
