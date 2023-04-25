#include <iostream>
#include "include\tensor_impl.h"
#include "include\tensor.h"
using namespace SimpleTensor;
//TensorImpl MakeMatrix(index_t n, index_t m) {
//    TensorImpl tensor({n, m});
//    for (index_t i = 0; i < n; ++i)
//        for (index_t j = 0; j < m; ++j)
//            tensor[{i, j}] = i*m+j;
//    return tensor;
//}
Tensor MakeMatrix(index_t n, index_t m) {
	Tensor tensor({n, m});
	for (index_t i = 0; i < n; ++i)
		for (index_t j = 0; j < m; ++j)
			tensor[{i, j}] = i*m+j;
	return tensor;
}
int main() {

    Tensor tensor = MakeMatrix(3, 4);
    std::cout << tensor << std::endl;
//    auto ptr = tensor.view({2, 4});
//    std::cout << *ptr << std::endl;
    auto ptr = tensor.transpose(0, 1);
    std::cout << ptr << std::endl;
//    std::cout << tensor.size() << std::endl;
    return 0;
}
