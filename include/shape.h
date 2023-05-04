#ifndef TENSOR_SHAPE_H
#define TENSOR_SHAPE_H

#include "array.h"
#include "allocator.h"

#include <initializer_list>

namespace st {
    using IndexArray = Array<index_t>;
    class Shape {
    public:
        Shape(std::initializer_list<index_t> dim);
        Shape(const Shape& other, index_t skip);
        Shape(index_t* dim, index_t n_dim);
        Shape(IndexArray &&dim);

        Shape(const Shape &dim) = default;
        Shape(Shape &&dim) = default;
        ~Shape() = default;

        [[nodiscard]] index_t d_size() const;
        [[nodiscard]] index_t sub_size(index_t start_dim, index_t end_dim) const;
        [[nodiscard]] index_t sub_size(index_t start_dim) const;
        bool operator==(const Shape &other) const;

        [[nodiscard]] index_t n_dim() const { return _dim.size(); }
        index_t& operator[](index_t idx) { return _dim[idx]; }
        index_t operator[](index_t idx) const { return _dim[idx]; }
        operator const IndexArray() const { return this->_dim; }
        friend std::ostream &operator<<(std::ostream &out, const Shape &sh);
    private:
        IndexArray _dim;
    };
} // SimpleTensor

#endif //TENSOR_SHAPE_H
