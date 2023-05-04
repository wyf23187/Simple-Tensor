#ifndef TENSOR_TENSOR_H
#define TENSOR_TENSOR_H

#include "tensor_impl.h"

namespace st {

    class Tensor
	{
	 public:
		//constructors
		Tensor(const Storage& storage, const Shape& shape, const IndexArray& stride);
		Tensor(const Storage& storage, const Shape& shape);
		explicit Tensor(const Shape& shape);
		Tensor(const data_t* data, const Shape& shape);
		Tensor(Storage&& storage, Shape&& shape, IndexArray&& stride);
		Tensor(const Tensor& other) = default;
		Tensor(Tensor&& other) = default;
		explicit Tensor(Alloc::NonTrivalUniquePtr<TensorImpl>&& ptr);

		//inline function
		[[nodiscard]] index_t n_dim() const { return _impl->n_dim(); }
		[[nodiscard]] index_t d_size() const { return  _impl->d_size(); }
		[[nodiscard]] index_t size(index_t idx) const { return _impl->size(idx); }
		[[nodiscard]] const Shape& size() const { return _impl->size(); }
		[[nodiscard]] index_t offset() const { return _impl->offset(); }
		[[nodiscard]] const IndexArray& stride() const { return _impl->stride(); }

		//methods
		[[nodiscard]] bool is_contiguous();
		[[nodiscard]] data_t item() const;
		[[nodiscard]] data_t item(const int idx) const;
		data_t &operator[](std::initializer_list<index_t> dims);
		data_t operator[](std::initializer_list<index_t> dims) const;

		[[nodiscard]] Tensor slice(index_t idx, index_t dim = 0) const;
		[[nodiscard]] Tensor slice(index_t start, index_t end, index_t dim = 0) const;
		[[nodiscard]] Tensor transpose(index_t dim1, index_t dim2) const;
		[[nodiscard]] Tensor permute(std::initializer_list<index_t> dims) const;

		//friend function
		friend std::ostream& operator<<(std::ostream& out, const Tensor& tensor);



	 private:
	 	Alloc::NonTrivalUniquePtr<TensorImpl> _impl;
    };

} // SimpleTensor

#endif //TENSOR_TENSOR_H
