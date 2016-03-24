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
	

	cuda_add_executable(sr src/main.cu src/helper.cu src/helper.h)

	target_link_libraries(sr ${CUDA_LIBARIES} ${OpenCV_LIBS})
else()
	message(CRITICAL_ERROR "CUDA cannot be found, please ensure CUDA dependencies are installed. CMake will exit now.")
endif()



