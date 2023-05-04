// dynamic array using share_ptr or unique_ptr

#ifndef ARRAY_H
#define ARRAY_H

#include "allocator.h"
#include <memory>
#include <vector>
#include <initializer_list>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <iostream>

namespace st {
    // unique_ptr array
    template<typename DType>
    class Array {
    public:
        Array(index_t size) :
            size_(size), d_ptr(Alloc::unique_allocate<DType>(size_*sizeof(DType))) {}
        Array(std::initializer_list<DType> d_list) : Array(d_list.size()) {
            auto ptr = d_ptr.get();
            for (auto d : d_list) {
                *ptr = d;
                ++ptr;
            }
        }
        Array(std::vector<DType> d_list) : Array(d_list.size()) {
            auto ptr = d_ptr.get();
            for (auto d : d_list) {
                *ptr = d;
                ++ptr;
            }
        }
        Array(const Array<DType> &other) :
            size_(other.size()), d_ptr(Alloc::unique_allocate<DType>(size_*sizeof(DType))){
            std::memcpy(this->d_ptr.get(), other.d_ptr.get(), size_*sizeof(DType));
        }
        Array(const DType *arr, index_t size) :
            size_(size), d_ptr(Alloc::unique_allocate<DType>(size_*sizeof(DType))) {
            std::memcpy(this->d_ptr.get(), arr, size_*sizeof(DType));
        }
        explicit Array(Array<DType>&& other) = default;

        ~Array() = default;

        DType& operator[](index_t idx) { return d_ptr.get()[idx]; }
        DType operator[](index_t idx) const { return d_ptr.get()[idx]; }

        int size() const { return this->size_; }
        void memset(int value) const { std::memset(d_ptr.get(), value, size_*sizeof(DType));}
        void fill(DType value) const { std::fill_n(d_ptr.get(), size_, value); }

    private:
        index_t size_;
        Alloc::TrivalUniquePtr<DType> d_ptr;
    };
}

#endif //ARRAY_H
