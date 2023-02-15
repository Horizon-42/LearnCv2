cmake_minimum_required(VERSION 3.5)

project(Pores)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 寻找 OpenCV 库
find_package(OpenCV REQUIRED)

# 添加可执行文件
add_executable(PoresExe pores_detect.cpp)

# 链接 OpenCV 库
target_link_libraries(PoresExe ${OpenCV_LIBS})
