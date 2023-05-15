#ifndef TENSOR_TENSOR_IMPL_H
#define TENSOR_TENSOR_IMPL_H

#include "shape.h"
#include "storage.h"
#include "array.h"
#include "allocator.h"
#include "exp.h"

#include <initializer_list>

namespace st {
    class TensorImpl {
    public:
        // constructor
        TensorImpl(const Storage& Storage, const Shape& Shape, const IndexArray& stride);
        TensorImpl(const Storage& Storage, const Shape& Shape);
        explicit TensorImpl(const Shape& Shape);
        TensorImpl(const data_t* data, const Shape& Shape);
        TensorImpl(Storage&& Storage, Shape&& Shape, IndexArray&& stride);
        TensorImpl(const TensorImpl& other) = default;
        TensorImpl(TensorImpl&& other) = default;
        template<typename ImplType>
        explicit TensorImpl(const ImplType& impl) : TensorImpl(impl->size()) {
            this->operator=(impl);
        }

        // inline function
        [[nodiscard]] index_t n_dim() const { return _shape.n_dim(); }
        [[nodiscard]] index_t d_size() const { return  _shape.d_size(); }
        [[nodiscard]] index_t size(index_t idx) const { return _shape[idx]; }
        [[nodiscard]] const Shape& size() const { return _shape; }
        [[nodiscard]] index_t offset() const { return _storage.offset(); }
        [[nodiscard]] const IndexArray& stride() const { return _stride; }

        // methods
        bool is_contiguous();

        data_t& operator[](std::initializer_list<index_t> dims); // use initializer list to access/modify the data.
        data_t operator[](std::initializer_list<index_t> dims) const;
        [[nodiscard]] data_t item() const;
        [[nodiscard]] data_t item(index_t idx) const;
		[[nodiscard]] data_t& item(index_t idx);
        [[nodiscard]] data_t eval(IndexArray idx) const;

        [[nodiscard]] Alloc::NonTrivalUniquePtr<TensorImpl> slice(index_t idx, index_t dim = 0) const;
        [[nodiscard]] Alloc::NonTrivalUniquePtr<TensorImpl> slice(index_t start_idx, index_t end_idx, index_t dim) const;
        [[nodiscard]] Alloc::NonTrivalUniquePtr<TensorImpl> transpose(index_t dim1, index_t dim2) const;
        [[nodiscard]] Alloc::NonTrivalUniquePtr<TensorImpl> view(const Shape& Shape) const;
        [[nodiscard]] Alloc::NonTrivalUniquePtr<TensorImpl> permute(std::initializer_list<index_t> dims) const;

        // friend function
        friend std::ostream& operator<<(std::ostream& out, const TensorImpl& tensor);

        template<typename ImplType>
        TensorImpl& operator=(const ImplType& src) {
            std::vector<index_t> dim_cnt(n_dim());
            for (int i = 0; i < n_dim(); ++i) dim_cnt.push_back(0);
            int cnt = 0;
            while (cnt < d_size()) {
                int idx = 0;
                for (int i = 0; i < n_dim(); ++i) {
                    idx += dim_cnt[i] * _stride[i];
                }
                item(idx) = src->eval(dim_cnt);
                for (int i = n_dim()-1; i >= 0; --i) {
                    if (dim_cnt[i]+1 < _shape[i]) {
                        dim_cnt[i]++;
                        break;
                    } else {
                        dim_cnt[i] = 0;
                    }
                }
                ++cnt;
            }
            return *this;
        }

    protected:
        Storage _storage;
        Shape _shape;
        IndexArray _stride;
    };

} // SimpleTensor

#endif //TENSOR_TENSOR_IMPL_H
