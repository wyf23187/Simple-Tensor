#include "tensor.h"

namespace st
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
	Tensor Tensor::slice(index_t start, index_t end, index_t dim) const
	{
		return Tensor(_impl->slice(start, end, dim));
	}
	Tensor Tensor::view(const Shape& shape) const
	{
		return Tensor(_impl->view(shape));
	}
	Tensor Tensor::transpose(index_t dim1, index_t dim2) const
	{
		return Tensor(_impl->transpose(dim1, dim2));
	}
	Tensor Tensor::permute(std::initializer_list<index_t> dims) const
	{
		return Tensor(_impl->permute(dims));
	}
	std::ostream& operator<<(std::ostream& out, const Tensor& tensor)
	{
		out << *tensor._impl;
		return out;
	}


	//iterator
	Tensor::iterator::iterator(Tensor* tensor, std::vector<index_t> idx)
	{
		_tensor = tensor;
		_idx = std::move(idx);
		//_shape is a protected value in TensorImpl
		for(int i = 0; i < tensor->_impl->n_dim(); ++i)
			_idx.push_back(0);
	}

	Tensor::iterator& Tensor::iterator::operator++()
	{
		int cnt = 0;
		for (int i = (int)_idx.size()-1; i >= 0; --i)
		{
			if (_idx[i]+1 < _tensor->size()[i])
			{
				++_idx[i];
				break;
			}
			else
			{
				_idx[i] = 0;
				++cnt;
			}
		}
		if (cnt == _idx.size())
		{
			for (int i = 0; i < _idx.size(); ++i)
			{
				_idx[i] = _tensor->size()[i];
			}
		}
		return *this;
	}

	Tensor::iterator Tensor::iterator::operator++(int)
	{
		iterator tmp = *this;
		++*this;
		return tmp;
	}

	Tensor::iterator& Tensor::iterator::operator--()
	{
		for (int i = 0; i < _idx.size(); --i)
		{
			if (_idx[i] > 0)
			{
				--_idx[i];
				break;
			}
			else
			{
				_idx[i] = _tensor->size()[i]-1;
			}
		}
		return *this;
	}

	Tensor::iterator Tensor::iterator::operator--(int)
	{
		iterator tmp = *this;
		--*this;
		return tmp;
	}

	index_t Tensor::iterator::operator-(const iterator& other) const
	{
		index_t cnt = 0;
		std::vector<index_t> idx = _idx;
		while (idx != other._idx)
		{
			++cnt;
			for (int i = (int)idx.size()-1; i >= 0; --i)
			{
				if (idx[i] > 0)
				{
					--idx[i];
					break;
				}
				else
				{
					idx[i] = _tensor->size()[i]-1;
				}
			}
		}
		return cnt;
	}

	bool Tensor::iterator::operator==(const iterator& other) const
	{
		return _idx == other._idx && _tensor == other._tensor;
	}

	bool Tensor::iterator::operator!=(const iterator& other) const
	{
		return !(*this == other);
	}

	Tensor::iterator::reference Tensor::iterator::operator*() const
	{
		index_t idx = 0;
		for (int i = 0; i < _idx.size(); ++i)
		{
			idx += _idx[i] * _tensor->_impl->stride()[i];
		}
		idx += _tensor->_impl->offset();
		return _tensor->_impl->item(idx);
	}

	Tensor::iterator::pointer Tensor::iterator::operator->() const
	{
		return &**this;
	}

	//const_iterator
	//iterator_methods

	Tensor::iterator Tensor::begin()
	{
		return iterator(this, std::vector<index_t>());
	}

	Tensor::iterator Tensor::end()
	{
		std::vector<index_t> idx;
		for (int i = 0; i < _impl->n_dim(); ++i)
		{
			idx.push_back(_impl->size()[i]);
		}
		return iterator(this, idx);
	}

} // SimpleTensor