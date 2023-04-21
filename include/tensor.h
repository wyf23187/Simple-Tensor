#ifndef TENSOR
#define TENSOR

#include <iostream>
#include <vector>
#include <initializer_list>

namespace SimpleTensor {

class Tensor {
  public:
    // constructor
    Tensor();
    Tensor(std::initializer_list<int> dim_list);
    Tensor(double *arr, std::initializer_list<int> dim_list);

    // static method
    static Tensor randn(std::initializer_list<int> dim_list);
    static Tensor zeros(std::initializer_list<int> dim_list);

    // operator
    double &at(int idx, ...);
    double &item(int index);
    friend std::ostream &operator<<(std::ostream &out, Tensor t);
    friend Tensor operator+(const Tensor &A, const Tensor &B);
    friend Tensor operator*(const Tensor &A, double c);
    friend Tensor operator*(const Tensor &A, const Tensor &B);

  private:
    std::vector<int> dim;
    std::vector<double> value;
    int total_element;
    int ndim;
    void print(std::ostream &out, int i_dim, int idx);
};


}

#endif