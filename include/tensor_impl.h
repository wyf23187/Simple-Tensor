#ifndef TENSOR_TENSOR_IMPL_H
#define TENSOR_TENSOR_IMPL_H

#include "shape.h"
#include "storage.h"
#include "array.h"
#include "allocator.h"

#include <initializer_list>

namespace SimpleTensor {
    using IndexArray = Array<index_t>;
    class TensorImpl {
    public:
        // constructor
        TensorImpl(const Storage& storage, const Shape& shape, const IndexArray& stride);
        TensorImpl(const Storage& storage, const Shape& shape);
        explicit TensorImpl(const Shape& shape);
        TensorImpl(const data_t* data, const Shape& shape);
        TensorImpl(Storage&& storage, Shape&& shape, IndexArray&& stride);
        TensorImpl(const TensorImpl& other) = delete;
        TensorImpl(TensorImpl&& other) = default;

        // inline function
        index_t n_dim() const { return _shape.n_dim(); }
        index_t size(index_t idx) const { return _shape[idx]; }
        const Shape& size() const { return _shape; }
        index_t offset() const { return _storage.offset(); }
        const IndexArray& stride() const { return _stride; }

        // methods
        bool is_contiguous();

        data_t& operator[](std::initializer_list<index_t> dims); // use initializer list to access/modify the data.
        data_t operator[](std::initializer_list<index_t> dims) const;
        data_t item() const;

        Alloc::NonTrivalUniquePtr<TensorImpl> slice(index_t idx, index_t dim = 0) const;
        Alloc::NonTrivalUniquePtr<TensorImpl> slice(index_t start_idx, index_t end_idx, index_t dim) const;
        Alloc::NonTrivalUniquePtr<TensorImpl> transpose(index_t dim1, index_t dim2) const;
        Alloc::NonTrivalUniquePtr<TensorImpl> view(const Shape& shape) const;
        Alloc::NonTrivalUniquePtr<TensorImpl> permute(std::initializer_list<index_t> dims) const;

        // friend function
        friend std::ostream& operator<<(std::ostream& out, const TensorImpl& tensor);

    protected:
        Storage _storage;
        Shape _shape;
        Array<index_t> _stride;
    };

} // SimpleTensor

#endif //TENSOR_TENSOR_IMPL_H
