cmake_minimum_required(VERSION 3.10)

project(eff)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
# option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.7/bin/nvcc)
enable_language(CUDA)

# cuda env
include_directories(/usr/local/cuda-11.7/include)
link_directories(/usr/local/cuda-11.7/lib64)

# opencv env 
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

# tensorrt env
include_directories(/home/mw/TensorRT-8.6.1.6/include)
link_directories(/home/mw/TensorRT-8.6.1.6/lib)

# 告诉编译器 头文件这在里查找
include_directories(${PROJECT_SOURCE_DIR}/src/)
# 收集路径下的所有.cu文件和.cpp源文件，并以列表的形式存储在SRCS这个变量下
file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cu)

add_executable(${PROJECT_NAME} ${PROJECT_SOURCE_DIR}/main.cpp ${SRCS})
# target_link_libraries必须在add_executable之后，编译好可执行文件了才能链接上去
target_link_libraries(${PROJECT_NAME} cudart)
target_link_libraries(${PROJECT_NAME} nvinfer)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} -lpthread -lm)
# 编译的时候有可能会出现onnxparser未定义的error，加上这句即可
target_link_libraries(${PROJECT_NAME} nvonnxparser)


