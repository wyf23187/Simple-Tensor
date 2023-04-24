//
// Created by mix on 2023/4/24.
//

#include "Storage.h"

#include <cstring>

namespace SimpleTensor {
    Storage::Storage(index_t size) :
            size_(size), b_ptr(Alloc::shared_allocate<Data>(size*sizeof(data_t)+sizeof(index_t))), f_ptr(b_ptr->data_) {}
    Storage::Storage(const Storage &other, index_t offset) :
            size_(other.size_), b_ptr(other.b_ptr), f_ptr(other.f_ptr+offset) {}
    Storage::Storage(index_t size, data_t value) : Storage(size) {}
    Storage::Storage(const data_t *data, index_t size) : Storage(size) {
        std::memcpy(f_ptr, data, size*sizeof(data_t));
    }
} // SimpleTensor