cmake_minimum_required(VERSION 2.8)

project(ShapeRegistration)

find_package(CUDA REQUIRED)
if(CUDA_FOUND)
	find_package(OpenCV REQUIRED)
	include_directories(${OpenCV_INCLUDE_DIRS})
	#find_package(libpng REQUIRED)

	#specify binary name and source files to build

	set ( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --ptxas-options=-v --use_fast_math 
	--compiler-options -Wall -lopencvhighgui -lopencv_core )

	include(ExternalProject)
	ExternalProject_Add(
		lmfit-6.1
		SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/lmfit-6.1
		CONFIGURE_COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/lmfit-6.1/configure --prefix=${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/lmfit-6.1
		PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/lmfit-6.1
		BUILD_COMMAND make
		BUILD_IN_SOURCE 1
)
		
	

	cuda_add_executable(sr src/main.cu src/helper.cu src/helper.h) 
	target_link_libraries(sr ${CUDA_LIBARIES} ${OpenCV_LIBS})
else()
	message(CRITICAL_ERROR "CUDA cannot be found, please ensure CUDA dependencies are installed. CMake will exit now.")
endif()


