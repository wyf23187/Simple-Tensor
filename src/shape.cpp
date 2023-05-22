#include "shape.h"

#include <initializer_list>

namespace st {
    Shape::Shape(std::initializer_list<index_t> dim) : _dim(dim) {}
    Shape::Shape(const Shape& other, index_t skip) : _dim(other.n_dim() - 1) {
        // skip the [skip] element
        int idx = 0;
        while (idx < skip) {
            _dim[idx] = other._dim[idx];
            ++idx;
        }
        while (idx+1 < other._dim.size()) {
            _dim[idx] = other._dim[idx+1];
            ++idx;
        }
    }
    Shape::Shape(index_t *dim, index_t n_dim) : _dim(dim, n_dim) {}
    Shape::Shape(Array<index_t>&& dim) : _dim(std::move(dim)) {}

    index_t Shape::d_size() const {
        int size = 1;
        for (int i = 0; i < _dim.size(); ++i)
            size *= _dim[i];
        return size;
    }

    index_t Shape::sub_size(index_t start_dim, index_t end_dim) const {
        int size = 1;
        for (int i = start_dim; i < end_dim; ++i)
            size *= _dim[i];
        return size;
    }

    index_t Shape::sub_size(index_t start_dim) const {
        int size = 1;
        for (int i = start_dim; i < _dim.size(); ++i)
            size *= _dim[i];
        return size;
    }

    bool Shape::operator==(const Shape &other) const {
        if (this->n_dim() != other.n_dim()) return false;
        for (int i = 0; i < this->n_dim(); ++i) {
            if (_dim[i] != other._dim[i]) return false;
        }
        return true;
    }

    std::ostream& operator<<(std::ostream &out, const Shape &sh) {
        out << "(" << sh[0];
        for (int i = 1; i < sh.n_dim(); ++i)
            out << ", " << sh[i];
        out << ")";
        return out;
    }
} // SimpleTensor