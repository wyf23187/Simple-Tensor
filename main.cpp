#include <iostream>
#include "include\tensor_impl.h"
#include "include\tensor.h"
using namespace st;
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
			tensor[{i, j}] = i*m+j + 1;
	return tensor;
}
int main() {

	Tensor A = Tensor::zeros_like(MakeMatrix(3, 4));
	std::cout << A << std::endl;
    return 0;
}
