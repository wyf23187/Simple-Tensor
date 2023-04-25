#include "tensor_impl.h"

#include <memory>
#include <cmath>
#include <iomanip>

namespace SimpleTensor {
    // constructor
    TensorImpl::TensorImpl(const Storage& storage, const Shape& shape, const IndexArray& stride) :
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
        for (int i = 0; i < shape.d_size(); ++i)
            _storage[i] = 0;
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
        for (int i = 0; i < n_dim()+1; ++i) {
            if (_stride[i] != _shape.sub_size(i+1)) return false;
        }
        if (_stride[n_dim()-1] != 1) return false;
        return true;
    }

    data_t& TensorImpl::operator[](std::initializer_list<index_t> dims) {
        index_t index = 0, dim = 0;
        for (auto v : dims) {
            index += v*_stride[dim];
            ++dim;
        }
        return _storage[index];
    }
    data_t TensorImpl::operator[](std::initializer_list<index_t> dims) const {
        index_t index = 0, dim = 0;
        for (auto v : dims) {
            index += v*_stride[dim];
            ++dim;
        }
        return _storage[index];
    }

    data_t TensorImpl::item() const {
        return _storage[0];
    }

    data_t TensorImpl::item(const int idx) const {
        return _storage[idx];
    }

    Alloc::NonTrivalUniquePtr<TensorImpl>
    TensorImpl::slice(index_t idx, index_t dim) const {
        Alloc::NonTrivalUniquePtr<TensorImpl> ptr;
        ptr = Alloc::unique_construct<TensorImpl>(
                Storage(_storage, offset()+_stride[dim]*idx),
                _shape, _stride);
        ptr->_shape[dim] = 1;
        ptr->_stride[dim] = 0;
        return ptr;
    }

    Alloc::NonTrivalUniquePtr<TensorImpl>
    TensorImpl::transpose(index_t dim1, index_t dim2) const {
        Alloc::NonTrivalUniquePtr<TensorImpl> ptr;
        ptr = Alloc::unique_construct<TensorImpl>(_storage, _shape, _stride);
        std::swap(ptr->_shape[dim1], ptr->_shape[dim2]);
        std::swap(ptr->_stride[dim1], ptr->_stride[dim2]);
        return ptr;
    }

    Alloc::NonTrivalUniquePtr<TensorImpl>
    TensorImpl::view(const Shape &shape) const {
        Alloc::NonTrivalUniquePtr<TensorImpl> ptr;
        ptr = Alloc::unique_construct<TensorImpl>(_storage, shape);
        for (int i = 0; i < shape.n_dim(); ++i) {
            if (i+1 < shape.n_dim()) ptr->_stride[i] = shape.sub_size(i+1);
            else ptr->_stride[i] = 1;
        }
        return ptr;
    }

    Alloc::NonTrivalUniquePtr<TensorImpl>
    TensorImpl::permute(std::initializer_list<index_t> dims) const {
        Alloc::NonTrivalUniquePtr<TensorImpl> ptr;
        ptr = Alloc::unique_construct<TensorImpl>(_storage, _shape);
        int idx = 0;
        for (auto n_permute : dims) {
            ptr->_shape[idx] = _shape[n_permute];
            ptr->_stride[idx] = _stride[n_permute];
            ++idx;
        }
        return ptr;
    }

    // friend function
    std::ostream& operator<<(std::ostream& out, const TensorImpl& tensor) {
        int max_width = 0;
        for (int i = 0; i < tensor.d_size(); ++i) {
            int value = (int)std::abs(tensor.item(i));
            int dig = value = (int)(std::log10(value))+1;
            if (tensor.item(i) < 0) ++dig;
            max_width = std::max(max_width, dig);
        }
        int cnt = 0, idx = 0, end_flag = tensor.n_dim();
        std::vector<int> dim_cnt(tensor.n_dim());
        while (cnt < tensor.d_size()) {
            for (int i = 0; i < tensor.n_dim()-end_flag; ++i)
                out << " ";
            for (int i = 0; i < end_flag; ++i)
                out << "[";
            out << std::setw(max_width+4+1) << std::right << std::setprecision(4) << std::fixed;
            out << tensor.item(idx);
            end_flag = 0;
            for (int i = (int)tensor.n_dim()-1; i >= 0; --i) {
                if (dim_cnt[i]+1 < tensor.size()[i]) {
                    idx += tensor.stride()[i];
                    ++dim_cnt[i];
                    break;
                } else {
                    idx -= ((int)tensor.size()[i]-1)*tensor.stride()[i];
                    dim_cnt[i] = 0;
                    ++end_flag;
                }
            }
            if (end_flag == 0) out << ", ";
            else {
                for (int i = 0; i < end_flag; ++i) {
                    out << "]";
                }
                out << std::endl;

            }
            ++cnt;
        }
        return out;
    }

} // SimpleTensor
