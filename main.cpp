#include <iostream>
#include "include/tensor.h"
int main() {
    SimpleTensor::Tensor tensor = SimpleTensor::Tensor::randn({2, 2, 3});
    
    return 0;
}
