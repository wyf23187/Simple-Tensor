#ifndef TENSOR_EXP_H
#define TENSOR_EXP_H

#include "oper.h"
#include "storage.h"

namespace st {
    template<typename SubType>
    class Exp {
    public:
        inline const SubType& self() const {
            return *static_cast<const SubType*>(this);
        }
    };

    template<typename Op, typename LhsType, typename RhsType>
    class BinaryExp: public Exp<BinaryExp<Op, LhsType, RhsType>> {
        const LhsType& _lhs;
        const RhsType& _rhs;
    public:
        [[nodiscard]] inline data_t eval(IndexArray idx) const {
            return Op::eval(idx, _lhs, _rhs);
        }
        BinaryExp(const LhsType& _lhs, const RhsType& _rhs): _lhs(_lhs), _rhs(_rhs) {}
    };

    template<typename LhsType, typename RhsType>
    [[nodiscard]] inline BinaryExp<op::Add<LhsType, RhsType>, LhsType, RhsType> operator+(const LhsType& lhs, const RhsType& rhs) {
        return BinaryExp<op::Add<LhsType, RhsType>, LhsType, RhsType>(lhs, rhs);
    }

    template<typename LhsType, typename RhsType>
    [[nodiscard]] inline BinaryExp<op::Sub<LhsType, RhsType>, LhsType, RhsType> operator-(const LhsType& lhs, const RhsType& rhs) {
        return BinaryExp<op::Sub<LhsType, RhsType>, LhsType, RhsType>(lhs, rhs);
    }

    template<typename LhsType, typename RhsType>
    [[nodiscard]] inline BinaryExp<op::Mul<LhsType, RhsType>, LhsType, RhsType> operator*(const LhsType& lhs, const RhsType& rhs) {
        return BinaryExp<op::Mul<LhsType, RhsType>, LhsType, RhsType>(lhs, rhs);
    }

    template<typename LhsType, typename RhsType>
    [[nodiscard]] inline BinaryExp<op::Div<LhsType, RhsType>, LhsType, RhsType> operator/(const LhsType& lhs, const RhsType& rhs) {
        return BinaryExp<op::Div<LhsType, RhsType>, LhsType, RhsType>(lhs, rhs);
    }
}// SimpleTensor

#endif //TENSOR_EXP_H