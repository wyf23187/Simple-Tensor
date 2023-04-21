## SimpleTensor 实现

### Tensor 是什么

Tensor 可以理解为可变维数的数组，可以根据用户所需要的维数进行使用。但又不只是数组。它还能够进行一些运算操作。

我们的实现基本上是模拟 Pytorch 中的 Tensor 操作。

### Tensor 中数据的存储

为了方便存储数据，我们将所有高维数据压平到一维去存储，并辅助一个每一个维度大小的数组。

```cpp
class Tensor {
  private:
    std::vector<int> dim;
    std::vector<double> value;
    int total_elements;                 // value.size()
    int ndim;                           // dim.size()
}
```

### Tensor 的运算

#### 加法/减法

#### 数乘

#### 点乘

#### 矩阵乘法