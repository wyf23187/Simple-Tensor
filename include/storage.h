#ifndef TENSOR_STORAGE_H
#define TENSOR_STORAGE_H

#include "allocator.h"

namespace st {
    typedef double data_t;
    class Storage {
    public:
        explicit Storage(index_t size);
        Storage(const Storage& other, index_t offset);
        Storage(index_t size, data_t value);
        Storage(const data_t *data, index_t size);

        explicit Storage(const Storage& other) = default;
        explicit Storage(Storage&& other) = default;

        ~Storage() = default;

        Storage& operator=(const Storage& other) = delete;

        data_t operator[](index_t idx) const { return f_ptr[idx]; }
        data_t& operator[](index_t idx) { return f_ptr[idx]; }
        [[nodiscard]] index_t offset() const { return f_ptr - b_ptr->data_; }
        // index_t version() const { return b_ptr->version; }
        // void increment_version() { ++b_ptr->version; }
        index_t size_;
    private:
        struct Data {
            // index_t version_; // what is its meaning?
            data_t data_[1];
        };
        std::shared_ptr<Data> b_ptr; // base pointer
        data_t* f_ptr; // float pointer
    };

} // SimpleTensor

#endif //TENSOR_STORAGE_H
