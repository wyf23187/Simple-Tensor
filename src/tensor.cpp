#include "../include/tensor.h"

#include <cstdarg>
#include <iostream>
#include <random>
#include <iomanip>
#include <ctime>
#include <initializer_list>

namespace SimpleTensor {

Tensor::Tensor() {
    this->dim.clear();
    this->value.clear();
    this->total_element = 0;
    this->ndim = 0;
}

Tensor::Tensor(std::initializer_list<int> dim_list) {
    this->dim.clear();
    this->total_element = 1;
    for (auto d : dim_list) {
        this->dim.push_back(d);
        this->total_element *= d;
    }
    this->value.resize(this->total_element);
    this->ndim = this->dim.size();
    for (auto &v : this->value) {
        v = 0;
    }
}

// static method

Tensor Tensor::randn(std::initializer_list<int> dim_list) {
    std::default_random_engine rg;
    rg.seed(time(NULL));
    std::normal_distribution<double> dis(0, 1);
    Tensor t(dim_list);
    int tot = 1;
    for (auto d : dim_list) tot *= d;
    for (int idx = 0; idx < tot; ++idx) {
        t.item(idx) = dis(rg);
    }
    return t;
}

// operators

double &Tensor::at(int idx1, ...) {
    va_list idxs;
    va_start(idxs, idx1);
    int idx = idx1;
    if (this->ndim == 1) return this->value[idx1];
    idx = idx1*this->dim[1];
    for (int i = 1; i < this->ndim; ++i) {
        int now_index = va_arg(idxs, int);
        if (i+1 != this->ndim) idx += now_index*this->dim[i+1];
        else idx += now_index;
    }
    va_end(idxs);
    return this->value[idx];
}

double &Tensor::item(int index) {
    return this->value[index];
}

std::ostream &operator<<(std::ostream &out, Tensor t) {
    t.print(out, 0, 0);
    return out;
}

void Tensor::print(std::ostream &out, int i_dim, int idx) {
    for (int i = 0; i < this->dim[i_dim]; ++i) {
        if (i_dim+1 == this->ndim) {
            out << std::fixed << std::setprecision(4) << std::right << std::setw(7);
            out << this->value[idx+i] << ", ";
        } else {
            this->print(out, i_dim+1, idx+i*this->dim[i_dim+1]);
        }
    }
    if (i_dim == this->ndim-1 || i_dim == this->ndim-2)
        out << std::endl;
}

}