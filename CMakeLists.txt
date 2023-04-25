cmake_minimum_required(VERSION 3.10)
project(Tensor_)
include_directories(include/)
add_executable(tensor
        main.cpp
        src/tensor.cpp include/tensor.h
        src/Storage.cpp include/Storage.h
        src/Shape.cpp include/Shape.h
        src/allocator.cpp src/tensor_impl.cpp include/tensor_impl.h)