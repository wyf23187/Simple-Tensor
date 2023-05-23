#ifndef TENSOR_TENSOR_H
#define TENSOR_TENSOR_H

#include "tensor_impl.h"
#include "exp.h"
#include "oper.h"
#include "allocator.h"

namespace st {

    class Tensor : public Exp<TensorImpl>
	{
        using Exp<TensorImpl>::impl_ptr;
	 public:
		//constructors
		Tensor(const Storage& storage, const Shape& shape, const IndexArray& stride);
		Tensor(const Storage& storage, const Shape& shape);
		explicit Tensor(const Shape& shape);
		Tensor(const data_t* data, const Shape& shape);
		Tensor(Storage&& storage, Shape&& shape, IndexArray&& stride);
		Tensor(const Tensor& other) = default;
		Tensor(Tensor&& other) = default;
		Tensor& operator=(const Tensor &other)
        {
			if (this != &other) {
				impl_ptr = Alloc::unique_construct<TensorImpl>(*other.impl_ptr);
			}
			return *this;
		}
		Tensor& operator=(Tensor &&other) = default;
        ~Tensor() = default;
		explicit Tensor(Alloc::NonTrivalUniquePtr<TensorImpl>&& ptr);
        template<typename ImplType>
        Tensor(const Exp<ImplType>& impl) : Tensor(impl.ptr()->size())
        {
            impl_ptr->operator=(impl.ptr());
        }

		//inline function
		[[nodiscard]] index_t n_dim() const { return impl_ptr->n_dim(); }
		[[nodiscard]] index_t d_size() const { return  impl_ptr->d_size(); }
		[[nodiscard]] index_t size(index_t idx) const { return impl_ptr->size(idx); }
		[[nodiscard]] const Shape& size() const { return impl_ptr->size(); }
		[[nodiscard]] index_t offset() const { return impl_ptr->offset(); }
		[[nodiscard]] const IndexArray& stride() const { return impl_ptr->stride(); }

		//methods
		[[nodiscard]] bool is_contiguous();
		[[nodiscard]] data_t item() const;
		[[nodiscard]] data_t item(int idx) const;
		[[nodiscard]] data_t eval(IndexArray idx) const;
		data_t &operator[](std::initializer_list<index_t> dims);
		data_t operator[](std::initializer_list<index_t> dims) const;

		[[nodiscard]] Tensor slice(index_t idx, index_t dim = 0) const;
		[[nodiscard]] Tensor slice(index_t start, index_t end, index_t dim) const;
		[[nodiscard]] Tensor transpose(index_t dim1, index_t dim2) const;
		[[nodiscard]] Tensor view(const Shape& Shape) const;
		[[nodiscard]] Tensor permute(std::initializer_list<index_t> dims) const;
        [[nodiscard]] Tensor sum(int idx) const;

		//friend function
		friend std::ostream& operator<<(std::ostream& out, const Tensor& tensor);

		//iterator
		class const_iterator
		{
		 private:
			using const_reference = const data_t&;
			using const_pointer = const data_t*;
			std::vector<index_t> _idx;
			const Tensor* _tensor;
		 public:
			const_iterator(const Tensor* tensor,std::vector<index_t> idx);
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
			const_reference operator*() const;
			const_pointer operator->() const;
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

		template<typename ImplType>
		Tensor& operator=(const Exp<ImplType>& src_){
            impl_ptr->operator=(src_.ptr());
			return *this;
		}

        static Tensor ones(const Shape& shape);
        static Tensor ones_like(const Tensor& tensor);
        static Tensor zeros(const Shape& shape);
        static Tensor zeros_like(const Tensor& tensor);
        static Tensor rand(const Shape& shape);
        static Tensor rand_like(const Tensor& tensor);
        static Tensor randn(const Shape& shape);
        static Tensor randn_like(const Tensor& tensor);
        [[nodiscard]] data_t sum() const;
    };

} // st

#endif //TENSOR_TENSOR_H
