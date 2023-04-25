//
// Created by mix on 2023/4/24.
//

#include "tensor.h"

namespace SimpleTensor
{
	//constructors
	Tensor::Tensor(const Storage& storage, const Shape& shape, const IndexArray& stride) :
		_impl(Alloc::unique_construct<TensorImpl>(storage, shape, stride)) {}
	Tensor::Tensor(const Storage& storage, const Shape& shape) :
		_impl(Alloc::unique_construct<TensorImpl>(storage, shape)) {}
	Tensor::Tensor(const Shape& shape) :
		_impl(Alloc::unique_construct<TensorImpl>(shape)) {}
	Tensor::Tensor(const data_t* data, const Shape& shape) :
		_impl(Alloc::unique_construct<TensorImpl>(data, shape)) {}
	Tensor::Tensor(Storage&& storage, Shape&& shape, IndexArray&& stride) :
		_impl(Alloc::unique_construct<TensorImpl>(std::move(storage), std::move(shape), std::move(stride))) {}
	Tensor::Tensor(Alloc::NonTrivalUniquePtr<TensorImpl>&& ptr) : _impl(std::move(ptr)) {}

	//operations
	bool Tensor::is_contiguous() { return _impl->is_contiguous(); }
	data_t Tensor::item() const { return _impl->item(); }
	data_t Tensor::item(const int idx) const { return _impl->item(idx); }
	data_t &Tensor::operator[](std::initializer_list<index_t> dims) { return _impl->operator[](dims); }
	data_t Tensor::operator[](std::initializer_list<index_t> dims) const { return _impl->operator[](dims); }

	Tensor Tensor::slice(index_t idx, index_t dim) const
	{
		return Tensor(_impl->slice(idx, dim));
	}
//	Tensor Tensor::slice(index_t start, index_t end, index_t dim) const
//	{
//		return Tensor(_impl->slice(start, end, dim));
//	}
	Tensor Tensor::transpose(index_t dim1, index_t dim2) const
	{
		return Tensor(_impl->transpose(dim1, dim2));
	}
//	Tensor Tensor::permute(std::initializer_list<index_t> dims) const
//	{
//		return Tensor(_impl->permute(dims));
//	}
	std::ostream& operator<<(std::ostream& out, const Tensor& tensor)
	{
		out << *tensor._impl;
		return out;
	}
} // SimpleTensor