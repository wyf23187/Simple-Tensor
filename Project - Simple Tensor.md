# Project -> Simple Tensor

## 1.Project Introduction

### 1.1 The reason we implemented it

Tensor是一个多维数组，它可以用来表示在各种数学和科学领域中出现的各种对象。在机器学习和深度学习中，Tensor是非常重要的数据结构，因为它们可以用来表示神经网络中的输入、输出和参数.

> TensorFlow是一个流行的深度学习框架，它的名字就是由Tensor和Flow组合而来的，其中Tensor代表数据结构，Flow代表计算图中数据的流动。在TensorFlow中，所有的数据都被表示为Tensor，并且计算图中的每个操作都是对Tensor进行的。

例如，下面是一个使用Tensorflow的代码示例

```python
import tensorflow as tf

# 创建一个二维Tensor
a = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]], name='a')

# 转置操作
a_transpose = tf.transpose(a)

# 切片操作
a_slice = tf.slice(a, [1, 0], [2, 2])

# 创建一个会话并运行操作
with tf.Session() as sess:
    print("a:")
    print(sess.run(a))
    print("a_transpose:")
    print(sess.run(a_transpose))
    print("a_slice:")
    print(sess.run(a_slice))

```

作为深度学习的基础数据结构，我们希望能把这个只能在python中调用的数据结构在c++语言中实现。所以我们实现了一个底层的数据结构`Simple Tensor`，用来作为我们机器学习的起步尝试。

### 1.2 实现想法

为了实现这个 `Tensor`，我们的想法是将一个张量的 **形状**，实际存储的 **值** 和 **步长** 分开来存储。同时为了封装的完整性，将它的实现成一个指向实现的指针，即一个是 `TensorImpl` 类，一个是 `Tensor` 类，`Tensor` 类中有一个指向 `TensorImpl` 类的指针。这样的封装方便对这个 `Tensor` 进行扩展和修改，同时保证了内部的封闭性，用户只能看到外部的内容，不能看到和修改内部的具体实现。

为了实现懒计算，使用奇异递归模板，用模板类的方式去实现这个效果。

## 2.具体实现

### 2.1 Storage of data in Tensor

为了方便存储数据，我们将所有高维数据压平到一维去存储，并辅助一个形状数组，表示每一个维度大小。

#### 2.1.1 Data

由于数据的共享性，这里在 `allocateor.h` 和 `array.h` 里面实现了一个共享指针（`shared_ptr`）的数组，以此来实现数据存储的共享。

数据存储的声明和实现分别在 `storage.h` 和 `storage.cpp` 里，定义一个类 `Storage` 以表示数据的存储。

每个 `Storage` 里有两个指针，一个是以一个结构体为类型的共享的基指针 `b_ptr`，这个结构体里只有一个长度为 1 的数组，表示数据；一个是浮动的普通指针 `f_ptr`，用于访问这里面的数据。除此之外，还有一个 `offset` 表示这二者的偏差。通常情况下，二者应该是相等的，但是，经过某些操作变换之后，会导致二者不一样，同时 `offset` 也会不等于 0。

#### 2.1.2 Shape

同时，每一个张量形状是唯一的，所以用唯一指针（`unique_ptr`）实现这个数组。形状的声明和实现分别在 `shape.h` 与 `shape.cpp` 中，定义一个类 `Shape` 以表示形状。

#### 2.1.3 Stride

为了能够更加方便的遍历、处理和输出这个张量，额外引入一个步长的概念。步长是对每一维而言，将给出的多维坐标的形式转化为一维的内存中的位置。

在构造的时候，第 $i$ 维的步长就是 $i+1$ 维到第 $k$ 维的大小乘积，即

$$
stide_i = \prod _{j = i+1}^kshape_j
$$

最后一维第 $k$ 维的步长为 $1$。这样，对于一个 $k$ 维坐标 $(a_1,a_2,\cdots,a_k)$ 对应到的一维即为 
$$
\begin{split}
x &= a_1\times stride_1+a_2\times stride_2+\cdots+a_{k-1}\times stride_{k-1}+a_k\times stride_k\\
&= \prod_{i = 1}^ka_i\times stride_i
\end{split}
$$

### 2.2 Tensor 的基本运算

#### 2.2.1 Transpose

在Tensor中，转置操作用于**交换张量的维度**。

具体实现代码：

```c++
Alloc::NonTrivalUniquePtr<TensorImpl>
TensorImpl::transpose(index_t dim1, index_t dim2) const {
    CHECK_IN_RANGE(dim1, 0, n_dim(),
        "Dimension out of range (expected to be in range of [0, %d), but got %d)",
        n_dim(), dim1);
    CHECK_IN_RANGE(dim2, 0, n_dim(),
        "Dimension out of range (expected to be in range of [0, %d), but got %d)",
        n_dim(), dim2);
    Alloc::NonTrivalUniquePtr<TensorImpl> ptr;
    ptr = Alloc::unique_construct<TensorImpl>(_storage, _shape, _stride);
    std::swap(ptr->_shape[dim1], ptr->_shape[dim2]);
    std::swap(ptr->_stride[dim1], ptr->_stride[dim2]);
    return ptr;
}
```

#### 2.2.2 View

在Tensor中，View操作用于创建一个新的张量，该张量与原始张量共享相同的数据存储，但具有不同的形状和步幅。**这允许用户以不同的方式查看和操作张量，而无需复制原始数据。**

具体实现代码：

```c++
Alloc::NonTrivalUniquePtr<TensorImpl>
TensorImpl::view(const Shape &shape) const {
        CHECK_TRUE(is_contiguous(),
            "view() is only supported to contiguous tensor");
        CHECK_EQUAL(shape.d_size(), shape.d_size(),
            "Shape of size %d is invalid for input tensor with size %d",
            shape.d_size(), shape.d_size());
    Alloc::NonTrivalUniquePtr<TensorImpl> ptr;
    ptr = Alloc::unique_construct<TensorImpl>(_storage, shape);
    for (int i = 0; i < shape.n_dim(); ++i) {
        if (i+1 < shape.n_dim()) ptr->_stride[i] = shape.sub_size(i+1);
        else ptr->_stride[i] = 1;
        if (shape[i] == 1) ptr->_stride[i] = 0;
    }
    return ptr;
}
```

#### 2.2.3 Permute

在Tensor中，permute操作用于重新排列张量的维度。这允许用户以不同的顺序查看和操作张量，而无需复制原始数据。permute操作返回一个新的张量，其维度顺序与原始张量的维度顺序不同。

具体实现代码：

```c++
Alloc::NonTrivalUniquePtr<TensorImpl>
TensorImpl::permute(std::initializer_list<index_t> dims) const {
    CHECK_EQUAL(dims.size(), n_dim(),
        "Dimension not match (expected dims of %d, but got %zu)",
        n_dim(), dims.size());
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
```

#### 2.2.4 Slice

在Tensor中，slice操作用于从张量中选择一个子集。slice操作返回一个新的张量，其数据存储与原始张量相同，但形状和步幅可能不同。新张量的数据存储是原始张量中的一个子集，由start_idx和end_idx参数指定。

具体实现代码：

```c++
Alloc::NonTrivalUniquePtr<TensorImpl>
TensorImpl::slice(index_t start_idx, index_t end_idx, index_t dim) const {
    CHECK_IN_RANGE(dim, 0, n_dim(),
        "Dimension out of range (expected to be in range of [0, %d), but got %d)",
        n_dim(), dim);
    CHECK_IN_RANGE(start_idx, 0, size(dim),
        "Index %d is out of bound for dimension %d with size %d",
        start_idx, dim, size(dim));
    CHECK_IN_RANGE(end_idx, 0, size(dim)+1,
        "Range end %d is out of bound for dimension %d with size %d",
        end_idx, dim, size(dim));
    Alloc::NonTrivalUniquePtr<TensorImpl> ptr;
    ptr = Alloc::unique_construct<TensorImpl>(
            Storage(_storage, offset() + start_idx * _stride[dim]),
            _shape, _stride);
    ptr->_shape[dim] = end_idx-start_idx;
    return ptr;
}
```

### 2.3 Tensor 的数值运算

#### 2.3.0 基础实现

为了实现 `lazy calculation`，即一个表达式在被需要的时候才回去进行计算，将 `Tensor` 继承自一个 `Exp` 模板类，这个类的具体实现如下：

```cpp
template<typename SubType>
class Exp {
public:
    Exp(std::shared_ptr<SubType>&& ptr) : impl_ptr(std::move(ptr)) {}
    inline const SubType& self() const {
        return *impl_ptr;
    }
    inline const std::shared_ptr<SubType>& ptr() const {
        return impl_ptr;
    }
protected:
    std::shared_ptr<SubType> impl_ptr;
};
```

这之后，将所有运算分为双目运算和单目运算，分成两个类写：

```cpp
template<typename Op, typename LhsType, typename RhsType>
class BinaryExp { // Binary Expression
    public:
    [[nodiscard]] inline data_t eval(IndexArray idx) const {
        return Op::eval(idx, lhs_ptr, rhs_ptr);
    }
    BinaryExp(const std::shared_ptr<LhsType>& _lhs, const std::shared_ptr<RhsType> _rhs)
        :lhs_ptr(_lhs), rhs_ptr(_rhs) {}
    [[nodiscard]] Shape size() const {
        return Op::size(lhs_ptr, rhs_ptr);
    }
    [[nodiscard]] index_t size(index_t idx) const {
        if (idx >= lhs_ptr->n_dim()) return rhs_ptr->size(idx);
        if (idx >= rhs_ptr->n_dim()) return lhs_ptr->size(idx);
        return std::max(lhs_ptr->size(idx), rhs_ptr->size(idx));
    }
    [[nodiscard]] index_t n_dim() const {
        return std::max(lhs_ptr->n_dim(), rhs_ptr->n_dim());
    }
    ~BinaryExp() = default;
    private:
    std::shared_ptr<LhsType> lhs_ptr;
    std::shared_ptr<RhsType> rhs_ptr;
};
template<typename Op, typename LhsType>
class UnaryExp { // Unary Expression
    public:
    [[nodiscard]] inline data_t eval(IndexArray idx) const {
        return Op::eval(idx, lhs_ptr);
    }
    UnaryExp(const std::shared_ptr<LhsType>&& ptr): lhs_ptr(ptr) {}
    [[nodiscard]] Shape& size() const {
        return lhs_ptr->size();
    }
    [[nodiscard]] index_t size(index_t idx) const {
        return lhs_ptr->size(idx);
    }
    [[nodiscard]] index_t n_dim() const {
        return lhs_ptr->n_dim();
    }
    private:
    std::shared_ptr<LhsType> lhs_ptr;
};
```

这样的操作后，之后的所有运算就是写一个对应操作的类，里面的成员函数都是静态成员函数。同时对对应操作符进行重载。

这里只是将表达式通过类模板存储下来，它的类型就可以判断出这个计算过程，这里面并没有计算，只有在赋值到一个 `Tensor` 的时候才进行计算。具体如下：

```cpp
template<typename ImplType>
TensorImpl& operator=(const ImplType& src) {
    std::vector<index_t> dim_cnt(n_dim(), 0);
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
```

对上述的具体分析，在下面详细讨论。

具体可以参考这个 [tutorial](https://github.com/dmlc/mshadow/tree/master/guide/exp-template)。

同时，操作要支持 Broadcast，即两个虽然 Shape 不同的张量，但是可以将某一个复制若干个打到二者可以运算，如：

```cpp
st::Tensor A = st::Tensor::rand({2, 3, 4});
st::Tensor B = st::Tensor::rand({3, 4});
st::Tensor C = A + B;
```

这里就会将 `B` 复制 2 份，使得二者可以运算，结果 `C` 是一个 `(2, 3, 4)` 的张量。

同时，若两个张量右对齐后，某个维度上大小不同，而其中一个大小为 1，也可以进行 Broadcast。

#### 2.3.1 加法/减法/乘法/除法

```cpp
struct Add {
    template<typename LhsType, typename RhsType>
    static data_t eval(IndexArray& idx, std::shared_ptr<LhsType> lhs, std::shared_ptr<RhsType> rhs) {
        CHECK_EXP_BROADCAST(lhs, rhs);
        return lhs->eval(idx)+rhs->eval(idx);
    }
    template<typename LhsType, typename RhsType>
    static Shape size(const std::shared_ptr<LhsType>& lhs, const std::shared_ptr<RhsType>& rhs) {
        return lhs->size();
    }
};
```

这样就可以将操作符存储在类型中。

```cpp
template<typename LhsType, typename RhsType>
[[nodiscard]] inline Exp<BinaryExp<op::Add, LhsType, RhsType>> operator+(const Exp<LhsType>& lhs, const Exp<RhsType>& rhs) {
    return Exp<BinaryExp<op::Add, LhsType, RhsType>>(
        std::make_shared<BinaryExp<op::Add, LhsType, RhsType>>(lhs.ptr(), rhs.ptr())
    );
}
```

减法、乘法、除法类似。

这里乘除都是对应位置的乘除。

#### 2.3.2 数乘

数乘可以看为一个大小为 `(1)` 的张量和另一个张量做乘法，然后在应用 Broadcast。只需对乘法进行重载即可。

```cpp
template<typename RhsType>
[[nodiscard]] inline Exp<BinaryExp<op::Mul, TensorImpl, RhsType>> operator*(data_t lhs_value, const Exp<RhsType>& rhs) {
    auto lhs = Exp<TensorImpl>(std::make_shared<TensorImpl>(Storage(1, lhs_value), Shape({1})));
    return Exp<BinaryExp<op::Mul, TensorImpl, RhsType>>(
        std::make_shared<BinaryExp<op::Mul, TensorImpl, RhsType>>(lhs.ptr(), rhs.ptr())
    );
}
```

#### 2.4.4 矩阵乘法

矩阵乘法略微复杂，但是通过对每一个位置依次求解，仍然可以做到。

```cpp
struct MatrixMul {
    template<typename LhsType, typename RhsType>
    static data_t eval(IndexArray& idx, std::shared_ptr<LhsType> lhs, std::shared_ptr<RhsType> rhs) {
        int l0, l1;
        l0 = lhs->size()[lhs->n_dim()-2];
        l1 = lhs->size()[lhs->n_dim()-1];
        int r0, r1;
        r0 = rhs->size()[rhs->n_dim()-2];
        r1 = rhs->size()[rhs->n_dim()-1];
        data_t res = 0;
        CHECK_EQUAL(l1, r0,
                    "mat1 and mat2 shapes cannot be multiplied (%dx%d and %dx%d)", l0, l1, r0, r1);
        for (int i = 0; i < l1; ++i) {
            IndexArray lidx = idx;
            IndexArray ridx = idx;
            lidx[idx.size()-1] = i;
            ridx[idx.size()-2] = i;
            res += lhs->eval(lidx)*rhs->eval(ridx);
        }
        return res;
    }
}
```

同时也实现了 Broadcast。矩阵乘法的 Broadcast 是只进行后两个维度的矩阵乘法，前面进行复制。

### 2.4 Tensor 的构造

在 `st::TensorMaker` 类里面实现，同时在 `Tensor` 类中调用这些静态函数。

```cpp
struct TensorMaker {
    static TensorImpl ones(const Shape& shape);
    static TensorImpl ones_like(const TensorImpl& tensor);
    static TensorImpl zeros(const Shape& shape);
    static TensorImpl zeros_like(const TensorImpl& tensor);
    static TensorImpl rand(const Shape& shape);
    static TensorImpl rand_like(const TensorImpl& tensor);
    static TensorImpl randn(const Shape& shape);
    static TensorImpl randn_like(const TensorImpl& tensor);
};
```

## 2. Our Highlights

### 2.1 Curiously Recurring Template Pattern(CRTP)

in exp.h:

```c++
#ifndef TENSOR_EXP_H
#define TENSOR_EXP_H

#include "storage.h"

namespace st {
    template<typename SubType>
    class Exp {
    public:
        Exp(std::shared_ptr<SubType>&& ptr) : impl_ptr(std::move(ptr)) {}
        inline const SubType& self() const {
            return *impl_ptr;
        }
        inline const std::shared_ptr<SubType>& ptr() const {
            return impl_ptr;
        }
    protected:
        std::shared_ptr<SubType> impl_ptr;
    };

    template<typename Op, typename LhsType, typename RhsType>
    class BinaryExp { // Binary Expression
    public:
        [[nodiscard]] inline data_t eval(IndexArray idx) const {
            return Op::eval(idx, lhs_ptr, rhs_ptr);
        }
        BinaryExp(const std::shared_ptr<LhsType>& _lhs, const std::shared_ptr<RhsType> _rhs)
            :lhs_ptr(_lhs), rhs_ptr(_rhs) {}
        [[nodiscard]] Shape size() const {
            return Op::size(lhs_ptr, rhs_ptr);
        }
        [[nodiscard]] index_t size(index_t idx) const {
            if (idx >= lhs_ptr->ndim()) return rhs_ptr->size(idx);
            if (idx >= rhs_ptr->ndim()) return lhs_ptr->size(idx);
            return max(lhs_ptr->size(idx), rhs_ptr->size(idx));
        }
        [[nodiscard]] index_t n_dim() const {
            return std::max(lhs_ptr->n_dim(), rhs_ptr->n_dim());
        }
        ~BinaryExp() = default;
    private:
        std::shared_ptr<LhsType> lhs_ptr;
        std::shared_ptr<RhsType> rhs_ptr;
    };

    template<typename Op, typename LhsType>
    class UnaryExp { // Unary Expression
    public:
        [[nodiscard]] inline data_t eval(IndexArray idx) const {
            return Op::eval(idx, lhs_ptr);
        }
        UnaryExp(const std::shared_ptr<LhsType>&& ptr): lhs_ptr(ptr) {}
        [[nodiscard]] Shape& size() const {
            return lhs_ptr->size();
        }
        [[nodiscard]] index_t size(index_t idx) const {
            return lhs_ptr->size(idx);
        }
        [[nodiscard]] index_t n_dim() const {
            return lhs_ptr->n_dim();
        }
    private:
        std::shared_ptr<LhsType> lhs_ptr;
    };
}// st

#endif //TENSOR_EXP_H
```

- 这段代码中使用了奇异递归模板
- 在这段代码中，`Exp` 类是一个模板类，它的模板参数是一个派生类的类型。这个派生类需要继承自 `Exp<SubType>`，并且需要提供 `eval` 和 `size` 函数的实现。这个派生类可以用来表示张量表达式中的一个子表达式。`BinaryExp` 和 `UnaryExp` 类都继承自 `Exp` 类，并且它们的模板参数都是一个派生类的类型。这些派生类需要提供 `eval` 和 `size` 函数的实现，这些函数将会在 `BinaryExp` 和 `UnaryExp` 类中被调用，用于计算张量表达式的值和形状。
- 通过使用 CRTP，我们可以在编译期间实现静态多态性，避免了在运行时进行虚函数调用的开销。此外，由于 CRTP 可以将子类的类型作为模板参数传递给父类，因此可以在编译期间进行类型检查，避免了运行时类型错误的风险。
- 此外，我们可以将 `BinaryExp` 和 `UnaryExp` 类的实现与具体的张量类型解耦，从而实现通用的张量表达式计算。这样，我们可以避免在每个张量类型中都写一遍表达式计算的代码，提高了代码的复用性和可维护性。

### 2.2 Pointer to Implementation（Pimpl）

Pimpl（指向实现的指针）是一种C++编程技术，用于隐藏类的实现细节，使得类的使用者无需了解其实现细节。这种技术也被称为“不透明指针”模式。

Pimpl的基本思想是将类的接口与其实现分离开来。不是在头文件中定义类的所有数据成员和成员函数，而是在头文件中只定义公共接口，并将实现细节移动到单独的源文件中。头文件只包含对实现类的指针，该类在源文件中定义。该指针用于访问类的实现细节。

在我们的代码中，我们在`tensor.h`中的**tensor**类之中，定义了一个指向**tensorimpl**类的指针**impl_ptr**

```c++
using Exp<TensorImpl>::impl_ptr;
```

例如我们在`tensorimpl.cpp`中实现了**eval**操作的代码

```c++
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
```

在`tensor.cpp`中就可以写成

```c++
data_t Tensor::eval(IndexArray idx) const
{
    return impl_ptr->eval(idx);
}
```

**使用Pimpl技术的优点：**

- 隐藏实现细节

> Pimpl技术可以将类的实现细节从类的用户中隐藏起来，使得用户只需要关注类的公共接口，而不需要了解其实现细节。这样可以避免用户直接访问类的私有成员和函数，从而提高了类的封装性。

- 减少编译依赖

> 使用Pimpl技术可以将类的实现细节放到单独的源文件中，这样当源文件的实现发生变化时，只需要重新编译源文件，而不需要重新编译包含该头文件的所有源文件。这可以减少编译依赖，从而提高编译速度。

- 提高二进制兼容性

> 由于Pimpl技术可以隐藏类的实现细节，所以当类的实现发生变化时，只需要重新编译实现文件，而不需要重新编译使用该类的源文件。这样可以提高二进制兼容性，避免由于类的实现变化导致的二进制不兼容问题。

- 提高代码的可维护性

> 使用Pimpl技术可以将类的实现细节与接口分离开来，使得代码更加清晰简洁，易于维护和修改。同时，Pimpl技术也可以避免由于类的实现变化而导致的代码修改，从而提高了代码的可维护性。

### 3. Smart pointer array

智能指针是C++程序员在管理动态内存时不可或缺的工具。然而，在某些情况下，当我们试图对数组执行动态内存管理时，我们会遇到一个问题：`shared_ptr`，默认情况下，不支持数组。这个限制背后的原因在于，`shared_ptr` 的默认删除器使用 `delete` 来删除智能指针内的对象。由于`delete`在使用`new`时需要一个单一指针，而在使用 `delete` 时需要一个数组类型，所以产生不匹配。

因此，我们想把数据存在智能指针数组之中，只能自己写一个智能指针数组的数据结构。（`虽然在c++17中已经支持了这样的操作，但不妨碍我们对它的实现`）

**我们的具体实现过程:**

在array.h文件中,定义了一个Array类模板,用于实现一个智能指针数组。它有以下几个特点:

1. 支持std::unique_ptr,使用Alloc::TrivalUniquePtr封装,负责数组的内存管理。

   ```c++
   Array(index_t size) : 
       size_(size), d_ptr(Alloc::unique_allocate<DType>(size_*sizeof(DType))) {}
   ```

2. 支持初始化列表和std::vector初始化。例如:

```c++
Array(std::initializer_list<DType> d_list) : Array(d_list.size()) {
    auto ptr = d_ptr.get();
    for (auto d : d_list) {
        *ptr = d;
        ++ptr; 
    }
}
Array(std::vector<DType> d_list) : Array(d_list.size()) {
    auto ptr = d_ptr.get();
    for (auto d : d_list) {
        *ptr = d;
        ++ptr; 
    } 
}
```

3. 支持拷贝构造函数。会重新分配内存并拷贝元素。

```c++
Array(const Array<DType> &other) :  
    size_(other.size()), d_ptr(Alloc::unique_allocate<DType>(size_*sizeof(DType))){
    std::memcpy(this->d_ptr.get(), other.d_ptr.get(), size_*sizeof(DType));
}
```

4. 支持移动构造函数。移动内存 ownership,不需要拷贝元素。

```c++
explicit Array(Array<DType>&& other) = default;
```

5. 重载[]操作符,用于访问元素。同时添加边界检查assert。

```c++
DType& operator[](index_t idx) { return d_ptr.get()[idx]; }  
DType operator[](index_t idx) const {
	assert(idx < size_);
	return d_ptr.get()[idx];
}
```

6. 提供size()、memset()和fill()方法,用于获取大小、设置所有元素值和填充元素。

```c++
int size() const { return this->size_; }
void memset(int value) const { std::memset(d_ptr.get(), value, size_*sizeof(DType));}  
void fill(DType value) const { std::fill_n(d_ptr.get(), size_, value); }
```

在allocator.h文件中,定义了一个Alloc类,用于管理Array类的内存分配与释放。它有以下特点:

1. 定义了trivial_delete_handler和nontrivial_delete_handler,用于Array的TrivalUniquePtr和NonTrivalUniquePtr使用。

```c++
class trivial_delete_handler {/*...*/};  
template<typename T>
class nontrivial_delete_handler {/*...*/};
```

   

2. 提供shared_allocate和unique_allocate方法分别分配共享内存和独占内存。返回相应的智能指针。

```c++
template<typename T>  
static std::shared_ptr<T> shared_allocate(index_t n_bytes) {/*...*/}
template<typename T>
static TrivalUniquePtr<T> unique_allocate(index_t n_bytes) {/*...*/}  
```

3. 提供shared_construct和unique_construct方法,可以在分配的内存上调用构造函数初始化对象。返回相应的智能指针。

```c++
template<typename T, typename... Args>  
static std::shared_ptr<T> shared_construct(Args&&...args) {/*...*/}
template<typename T, typename... Args>
static NonTrivalUniquePtr<T> unique_construct(Args&&...args) {/*...*/}
```

4. 使用一个std::multimap维护已分配的内存块,以实现内存重用,提高效率。

```c++
std::multimap<index_t, std::unique_ptr<void, free_deleter>> cache_; 
```

5. 实现Alloc类的单例,确保只有一个Alloc实例负责内存管理。

```c++
static Alloc& self();  
```

`array.h`中**Array**类模板与`allocator.h`中的**Alloc**类配合,实现了一个使用智能指针管理内存的动态数组,具有初始化、拷贝、移动等功能,是一种类型安全、异常安全的动态数组实现方式。