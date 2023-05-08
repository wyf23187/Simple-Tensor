cmake_minimum_required(VERSION 3.0)
project(Tensor)
add_executable(tensor
        main.cpp
        src/tensor.cpp
        src/allocator.cpp
        src/shape.cpp
        src/storage.cpp
        src/oper.cpp
        src/tensor_impl.cpp
        )
target_include_directories(tensor PUBLIC include)