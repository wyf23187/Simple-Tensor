cmake_minimum_required(VERSION 3.15)
project(Tensor)
set(CMAKE_CXX_STANDARD 20)

add_subdirectory(lib)
include_directories(googletest/include googletest)

add_executable(tensor
        main.cpp
        src/tensor.cpp
        src/allocator.cpp
        src/shape.cpp
        src/storage.cpp
        src/tensor_impl.cpp
        src/unit_test.cpp src/exception.cpp)
target_include_directories(tensor PUBLIC include)
target_link_libraries(tensor gtest gtest_main)