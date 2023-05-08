#ifndef TENSOR_TENSOR_H
#define TENSOR_TENSOR_H

#include "tensor_impl.h"

namespace st {

    class Tensor : public Exp<Tensor>
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
		Tensor& operator=(const Tensor &other) = default;
		Tensor& operator=(Tensor &&other) = default;
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
		[[nodiscard]] Tensor view(const Shape& Shape) const;
		[[nodiscard]] Tensor permute(std::initializer_list<index_t> dims) const;

		//friend function
		friend std::ostream& operator<<(std::ostream& out, const Tensor& tensor);

		//iterator
		class const_iterator
		{
		 public:
			const_iterator(const const_iterator& other) = default;
			const_iterator(const_iterator&& other) = default;
			~const_iterator() = default;

			const_iterator& operator=(const const_iterator& other) = default;
			const_iterator& operator=(const_iterator&& other) = default;
			const_iterator& operator++();
			const_iterator operator++(int);
			const_iterator& operator--();
			const_iterator operator--(int);
			index_t operator-(const const_iterator& other) const;
			bool operator==(const const_iterator& other) const;
			bool operator!=(const const_iterator& other) const;
			data_t operator*() const;
			data_t* operator->() const;
		};

		class iterator
		{
		 private:
			using reference = data_t&;
			using pointer = data_t*;
		 public:
			iterator(Tensor* tensor,std::vector<index_t> idx);
			iterator(const iterator& other) = default;
			iterator(iterator&& other) = default;
			~iterator() = default;

			iterator& operator=(const iterator& other) = default;
			iterator& operator=(iterator&& other) = default;

			iterator& operator++();
			iterator operator++(int);
			iterator& operator--();
			iterator operator--(int);

			index_t operator-(const iterator& other) const;
			bool operator==(const iterator& other) const;
			bool operator!=(const iterator& other) const;
			reference operator*() const;
			pointer operator->() const;
		 private:
			std::vector<index_t> _idx;
			Tensor* _tensor;
		};

		[[nodiscard]] const_iterator begin() const;
		[[nodiscard]] const_iterator end() const;
		[[nodiscard]] iterator begin();
		[[nodiscard]] iterator end();

		template<typename Etype>
		Tensor& operator=(const Exp<Etype>& src_){
			const Etype& src = src_.self();
			std::vector<index_t> dim_cnt;
			for (int i = 0; i < n_dim(); ++i) dim_cnt.push_back(0);
			int cnt = 0;
			while (cnt < d_size()) {
				int idx = 0;
				for (int i = 0; i < n_dim(); ++i) {
					idx += dim_cnt[i] * _impl->stride()[i];
				}
				_impl->item(idx) = src.eval(dim_cnt);
				for (int i = n_dim()-1; i >= 0; --i) {
					if (dim_cnt[i]+1 < size(i)) {
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
	 private:
	 	Alloc::NonTrivalUniquePtr<TensorImpl> _impl;
    };

} // SimpleTensor

#endif //TENSOR_TENSOR_H
