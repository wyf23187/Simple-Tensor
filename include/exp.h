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
            return Op::eval(lhs_ptr->eval(idx), rhs_ptr->eval(idx));
        }
        BinaryExp(const std::shared_ptr<LhsType>& _lhs, const std::shared_ptr<RhsType> _rhs)
            :lhs_ptr(_lhs), rhs_ptr(_rhs) {}
        [[nodiscard]] const Shape& size() const {
            return lhs_ptr->size();
        }
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
        [[nodiscard]] IndexArray& size() const {
            return lhs_ptr->size();
        }
    private:
        std::shared_ptr<LhsType> lhs_ptr;
    };
}// st

#endif //TENSOR_EXP_H