#include "tensor_impl.h"

#include <memory>

namespace SimpleTensor {
    // constructor
    TensorImpl::TensorImpl(const Storage& storage, const Shape& shape,
                           const IndexArray& stride) :
                           _storage(storage), _shape(shape), _stride(stride) {}
    TensorImpl::TensorImpl(const Storage& storage, const Shape& shape) :
        _storage(storage), _shape(shape), _stride(shape.n_dim()){
        for (int i = 0; i < shape.n_dim(); ++i) {
            if (i == shape.n_dim()-1) _stride[i] = 1;
            else _stride[i] = shape.sub_size(i+1);
        }
    }
    TensorImpl::TensorImpl(const Shape& shape) :
        _storage(shape.d_size()), _shape(shape), _stride(shape.n_dim()) {
        for (int i = 0; i < shape.n_dim(); ++i) {
            if (i == shape.n_dim()-1) _stride[i] = 1;
            else _stride[i] = shape.sub_size(i+1);
        }
    }
    TensorImpl::TensorImpl(const data_t* data, const Shape& shape) :
            _storage(shape.d_size()), _shape(shape), _stride(shape.n_dim()) {
        for (int i = 0; i < shape.d_size(); ++i)
            _storage[i] = data[i];
        for (int i = 0; i < shape.n_dim(); ++i) {
            if (i == shape.n_dim()-1) _stride[i] = 1;
            else _stride[i] = shape.sub_size(i+1);
        }
    }
    TensorImpl::TensorImpl(Storage&& storage, Shape&& shape, IndexArray&& stride) :
        _storage(std::move(storage)), _shape(std::move(shape)), _stride(std::move(stride)) {}

    // method
    bool TensorImpl::is_contiguous() {
        return true;
    }

    data_t& TensorImpl::operator[](std::initializer_list<index_t> dims) {
        index_t index = 0, dim = 0;
        for (auto v : dims) {
            index += v*_shape[dim];
            ++dim;
        }
        return _storage[index];
    }
    data_t TensorImpl::operator[](std::initializer_list<index_t> dims) const {
        index_t index = 0, dim = 0;
        for (auto v : dims) {
            index += v*_shape[dim];
            ++dim;
        }
        return _storage[index];
    }

    data_t TensorImpl::item() const {
        return _storage[0];
    }

} // SimpleTensor
