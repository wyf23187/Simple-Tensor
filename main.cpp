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
	Tensor t1 = MakeMatrix(3, 4);
	Tensor t2 = MakeMatrix(3, 4);
	Tensor t3 = MakeMatrix(3, 4);
//	std::cout << t1 << std::endl;
//	std::cout << t2 << std::endl;
	BinaryExp t4 = t1+t2;
	BinaryExp t5 = t1*t2;
//	t2 = t4; 有问题
	Tensor::iterator it = {&t1, {}};
	std::cout << *(++it) << std::endl;
	std::cout << *(++it) << std::endl;
	std::cout << *t1.begin() << std::endl;
	std::cout << t1 << std::endl;
//	std::cout << *t1.end() << std::endl; 有问题

	Tensor x = t1.slice(0);
  	std::cout << x << std::endl;
  	auto x1 = t1.transpose(0, 1);
  	std::cout << x1 << std::endl;
  	auto x2 = t1.view({1, 12});
  	std::cout << x2 << std::endl;
    return 0;
}
