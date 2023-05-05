#include "tensor_impl.h"

#include <memory>
#include <cmath>
#include <iomanip>

namespace st {
    // constructor
    TensorImpl::TensorImpl(const Storage& storage, const Shape& shape, const IndexArray& stride) :
        _storage(storage), _shape(shape), _stride(stride) {}
    TensorImpl::TensorImpl(const Storage& storage, const Shape& shape) :
        _storage(storage), _shape(shape), _stride(shape.n_dim()){
        for (int i = 0; i < shape.n_dim(); ++i) {
            if (i == shape.n_dim()-1) _stride[i] = 1;
            else _stride[i] = shape.sub_size(i+1);
            if (shape[i] == 1) _stride[i] = 0;
        }
    }
    TensorImpl::TensorImpl(const Shape& shape) :
        _storage(shape.d_size()), _shape(shape), _stride(shape.n_dim()) {
        for (int i = 0; i < shape.d_size(); ++i)
            _storage[i] = 0;
        for (int i = 0; i < shape.n_dim(); ++i) {
            if (i == shape.n_dim()-1) _stride[i] = 1;
            else _stride[i] = shape.sub_size(i+1);
            if (shape[i] == 1) _stride[i] = 0;
        }
    }
    TensorImpl::TensorImpl(const data_t* data, const Shape& shape) :
        _storage(shape.d_size()), _shape(shape), _stride(shape.n_dim()) {
        for (int i = 0; i < shape.d_size(); ++i)
            _storage[i] = data[i];
        for (int i = 0; i < shape.n_dim(); ++i) {
            if (i == shape.n_dim()-1) _stride[i] = 1;
            else _stride[i] = shape.sub_size(i+1);
            if (shape[i] == 1) _stride[i] = 0;
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

    data_t TensorImpl::item(int idx) const {
        return _storage[idx+offset()];
    }

    data_t TensorImpl::eval(IndexArray idx) const {
        int index = 0;
        if (idx.size() >= _shape.n_dim()) {
            for (int i = idx.size() - n_dim(); i < idx.size(); ++i)
                index += idx[i]*_stride[i-(idx.size()-n_dim())];
        } else {
            for (int i = 0; i < idx.size(); ++i)
                index += idx[i]*_stride[i+(n_dim()-idx.size())];
        }
        return item(index);
    }

    Alloc::NonTrivalUniquePtr<TensorImpl>
    TensorImpl::slice(index_t idx, index_t dim) const {
        Alloc::NonTrivalUniquePtr<TensorImpl> ptr;
        ptr = Alloc::unique_construct<TensorImpl>(
                Storage(_storage, offset() + _stride[dim] * idx),
                _shape, _stride);
        ptr->_shape[dim] = 1;
        ptr->_stride[dim] = 0;
        return ptr;
    }

    Alloc::NonTrivalUniquePtr<TensorImpl>
    TensorImpl::slice(index_t start_idx, index_t end_idx, index_t dim) const {
        Alloc::NonTrivalUniquePtr<TensorImpl> ptr;
        ptr = Alloc::unique_construct<TensorImpl>(
                Storage(_storage, offset() + start_idx * _stride[dim]),
                _shape, _stride);
        ptr->_shape[dim] = end_idx-start_idx;
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

    // iterator
    TensorImpl::iterator::iterator(TensorImpl *tensor, std::vector<index_t> idx)
        : _tensor(tensor), _idx(idx) {
        for (int i = 0; i < tensor->_shape.n_dim(); ++i) {
            idx.push_back(0);
        }
    }

    TensorImpl::iterator& TensorImpl::iterator::operator++() {
        int cnt = 0;
        for (int i = (int)_idx.size()-1; i >= 0; --i) {
            if (_idx[i]+1 < _tensor->size()[i]) {
                ++_idx[i];
                break;
            } else {
                _idx[i] = 0;
                ++cnt;
            }
        }
        if (cnt == _idx.size()) {
            for (int i = 0; i < _idx.size(); ++i) {
                _idx[i] = _tensor->size()[i];
            }
        }
        return *this;
    }

    TensorImpl::iterator TensorImpl::iterator::operator++(int) {
        iterator tmp = *this;
        ++*this;
        return tmp;
    }

    TensorImpl::iterator& TensorImpl::iterator::operator--() {
        for (int i = 0; i < _idx.size(); --i) {
            if (_idx[i] > 0) {
                --_idx[i];
                break;
            } else {
                _idx[i] = _tensor->size()[i]-1;
            }
        }
        return *this;
    }

    TensorImpl::iterator TensorImpl::iterator::operator--(int) {
        iterator tmp = *this;
        --*this;
        return tmp;
    }

    TensorImpl::iterator& TensorImpl::iterator::operator+=(index_t n) {
        for (int i = 0; i < n; ++i) {
            ++*this;
        }
        return *this;
    }

    TensorImpl::iterator& TensorImpl::iterator::operator-=(index_t n) {
        for (int i = 0; i < n; ++i) {
            --*this;
        }
        return *this;
    }

    TensorImpl::iterator TensorImpl::iterator::operator+(index_t n) const {
        iterator tmp = *this;
        tmp += n;
        return tmp;
    }

    TensorImpl::iterator TensorImpl::iterator::operator-(index_t n) const {
        iterator tmp = *this;
        tmp -= n;
        return tmp;
    }

    index_t TensorImpl::iterator::operator-(const iterator& rhs) const {
        index_t cnt = 0;
        std::vector<index_t> idx = _idx;
        while (idx != rhs._idx) {
            ++cnt;
            for (int i = (int)idx.size()-1; i >= 0; --i) {
                if (idx[i] > 0) {
                    --idx[i];
                    break;
                } else {
                    idx[i] = _tensor->size()[i]-1;
                }
            }
        }
        return cnt;
    }

    bool TensorImpl::iterator::operator==(const iterator& rhs) const {
        return _idx == rhs._idx && _tensor == rhs._tensor;
    }

    bool TensorImpl::iterator::operator!=(const iterator& rhs) const {
        return !(*this == rhs);
    }

    TensorImpl::iterator::reference TensorImpl::iterator::operator*() const {
        index_t idx = 0;
        for (int i = 0; i < _idx.size(); ++i) {
            idx += _idx[i] * _tensor->_stride[i];
        }
        idx += _tensor->offset();
        return _tensor->_storage[idx];
    }

    TensorImpl::iterator::pointer TensorImpl::iterator::operator->() const {
        return &**this;
    }

    // const_iterator
    // iterator method
    TensorImpl::iterator TensorImpl::begin() {
        std::vector<index_t> idx;
        for (int i = 0; i < n_dim(); ++i) {
            idx.push_back(0);
        }
        return {this, idx};
    }

    TensorImpl::iterator TensorImpl::end() {
        std::vector<index_t> idx;
        for (int i = 0; i < n_dim(); ++i) {
            idx.push_back(size(i));
        }
        return {this, idx};
    }
} // SimpleTensor