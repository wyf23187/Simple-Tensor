#include <memory>
#include "tensor.h"
#include "exp.h"

namespace st
{
	//constructors
	Tensor::Tensor(const Storage& storage, const Shape& shape, const IndexArray& stride) :
		Exp<TensorImpl>(Alloc::unique_construct<TensorImpl>(storage, shape, stride)) {}
	Tensor::Tensor(const Storage& storage, const Shape& shape) :
		Exp<TensorImpl>(Alloc::unique_construct<TensorImpl>(storage, shape)) {}
	Tensor::Tensor(const Shape& shape) :
        Exp<TensorImpl>(Alloc::unique_construct<TensorImpl>(shape)) {}
	Tensor::Tensor(const data_t* data, const Shape& shape) :
        Exp<TensorImpl>(Alloc::unique_construct<TensorImpl>(data, shape)) {}
	Tensor::Tensor(Storage&& storage, Shape&& shape, IndexArray&& stride) :
        Exp<TensorImpl>(Alloc::unique_construct<TensorImpl>(std::move(storage), std::move(shape), std::move(stride))) {}
	Tensor::Tensor(Alloc::NonTrivalUniquePtr<TensorImpl>&& ptr) : Exp<TensorImpl>(std::move(ptr)) {}

	//operations
	bool Tensor::is_contiguous() { return impl_ptr->is_contiguous(); }
	data_t Tensor::item() const { return impl_ptr->item(); }
	data_t Tensor::item(const int idx) const { return impl_ptr->item(idx); }
	data_t &Tensor::operator[](std::initializer_list<index_t> dims) { return impl_ptr->operator[](dims); }
	data_t Tensor::operator[](std::initializer_list<index_t> dims) const { return impl_ptr->operator[](dims); }

	Tensor Tensor::slice(index_t idx, index_t dim) const
	{
		return Tensor(impl_ptr->slice(idx, dim));
	}
	Tensor Tensor::slice(index_t start, index_t end, index_t dim) const
	{
		return Tensor(impl_ptr->slice(start, end, dim));
	}
	Tensor Tensor::view(const Shape& shape) const
	{
		return Tensor(impl_ptr->view(shape));
	}
	Tensor Tensor::transpose(index_t dim1, index_t dim2) const
	{
		return Tensor(impl_ptr->transpose(dim1, dim2));
	}
	Tensor Tensor::permute(std::initializer_list<index_t> dims) const
	{
		return Tensor(impl_ptr->permute(dims));
	}
	std::ostream& operator<<(std::ostream& out, const Tensor& tensor)
	{
		out << *tensor.impl_ptr;
		return out;
	}


	//iterator
	Tensor::iterator::iterator(Tensor* tensor, std::vector<index_t> idx)
	{
		_tensor = tensor;
		for (auto i : idx)
			_idx.push_back(i);
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
		if (*this == _tensor->end())
		{
			for (int i = 0; i < _idx.size(); ++i)
			{
				_idx[i] = _tensor->size()[i]-1;
			}
			return *this;
		}
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
			idx += _idx[i] * _tensor->impl_ptr->stride()[i];
		}
		return _tensor->impl_ptr->item(idx);
	}

	Tensor::iterator::pointer Tensor::iterator::operator->() const
	{
		return &**this;
	}

	//const_iterator
	Tensor::const_iterator::const_iterator(const Tensor* tensor, std::vector<index_t> idx)
	{
		_tensor = tensor;
		for (auto i : idx)
			_idx.push_back(i);
	}
	Tensor::const_iterator& Tensor::const_iterator::operator++()
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

	Tensor::const_iterator Tensor::const_iterator::operator++(int)
	{
		const_iterator tmp = *this;
		++*this;
		return tmp;
	}

	Tensor::const_iterator& Tensor::const_iterator::operator--()
	{
		if (*this == _tensor->end())
		{
			for (int i = 0; i < _idx.size(); ++i)
			{
				_idx[i] = _tensor->size()[i]-1;
			}
			return *this;
		}
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

	Tensor::const_iterator Tensor::const_iterator::operator--(int)
	{
		const_iterator tmp = *this;
		--*this;
		return tmp;
	}

	index_t Tensor::const_iterator::operator-(const const_iterator& other) const
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

	bool Tensor::const_iterator::operator==(const const_iterator& other) const
	{
		return _idx == other._idx && _tensor == other._tensor;
	}

	bool Tensor::const_iterator::operator!=(const const_iterator& other) const
	{
		return !(*this == other);
	}

	Tensor::const_iterator::const_reference Tensor::const_iterator::operator*() const
	{
		index_t idx = 0;
		for (int i = 0; i < _idx.size(); ++i)
		{
			idx += _idx[i] * _tensor->impl_ptr->stride()[i];
		}
		return _tensor->impl_ptr->item(idx);
	}

	Tensor::const_iterator::const_pointer Tensor::const_iterator::operator->() const
	{
		return &**this;
	}
	//iterator_methods

	Tensor::iterator Tensor::begin()
	{
		return iterator(this, std::vector<index_t>(impl_ptr->n_dim()));
	}

	Tensor::iterator Tensor::end()
	{
		std::vector<index_t> idx;
		for (int i = 0; i < impl_ptr->n_dim(); ++i)
		{
			idx.push_back(impl_ptr->size()[i]);
		}
		return iterator(this, idx);
	}

	Tensor::const_iterator Tensor::begin() const
	{
		return const_iterator(this, std::vector<index_t>(impl_ptr->n_dim()));
	}

	Tensor::const_iterator Tensor::end() const
	{
		std::vector<index_t> idx;
		for (int i = 0; i < impl_ptr->n_dim(); ++i)
		{
			idx.push_back(impl_ptr->size()[i]);
		}
		return const_iterator(this, idx);
	}
	data_t Tensor::eval(IndexArray idx) const
	{
		int index = 0;
		if (idx.size() >= impl_ptr->n_dim()) {
			for (int i = idx.size() - n_dim(); i < idx.size(); ++i)
				index += idx[i]*impl_ptr->stride()[i-(idx.size()-n_dim())];
		} else {
			for (int i = 0; i < idx.size(); ++i)
				index += idx[i]*impl_ptr->stride()[i+(n_dim()-idx.size())];
		}
		return item(index);
	}

    data_t Tensor::sum() const {
        return impl_ptr->sum();
    }

    Tensor Tensor::rand(const st::Shape &shape) {
        return Tensor(Alloc::unique_construct<TensorImpl>(TensorMaker::rand(shape)));
    }
    Tensor Tensor::ones(const st::Shape &shape) {
        return Tensor(Alloc::unique_construct<TensorImpl>(TensorMaker::ones(shape)));
    }
    Tensor Tensor::zeros(const st::Shape &shape) {
        return Tensor(Alloc::unique_construct<TensorImpl>(TensorMaker::zeros(shape)));
    }

} // SimpleTensor