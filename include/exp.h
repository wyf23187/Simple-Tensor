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
    class BinaryExp: public Exp<BinaryExp<Op, LhsType, RhsType>> { // Binary Expression
    public:
        [[nodiscard]] inline data_t eval(IndexArray idx) const {
            return Op::eval(idx, _lhs, _rhs);
        }
        BinaryExp(const LhsType& _lhs, const RhsType& _rhs): _lhs(_lhs), _rhs(_rhs) {}
    private:
        const LhsType& _lhs;
        const RhsType& _rhs;
    };

    template<typename Op, typename LhsType>
    class UnaryExp: public Exp<UnaryExp<Op, LhsType>> { // Unary Expression
    public:
        [[nodiscard]] inline data_t eval(IndexArray idx) const {
            return Op::eval(idx, _lhs);
        }
        UnaryExp(const LhsType& _lhs): _lhs(_lhs) {}
    private:
        const LhsType& _lhs;
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

    template<typename LhsType>
    [[nodiscard]] inline UnaryExp<op::Neg<LhsType>, LhsType> operator-(const LhsType& lhs) {
        return UnaryExp<op::Neg<LhsType>, LhsType>(lhs);
    }

    template<typename LhsType>
    [[nodiscard]] inline UnaryExp<op::Sin<LhsType>, LhsType> sin(const LhsType& lhs) {
        return UnaryExp<op::Sin<LhsType>, LhsType>(lhs);
    }

    template<typename LhsType>
    [[nodiscard]] inline UnaryExp<op::Cos<LhsType>, LhsType> cos(const LhsType& lhs) {
        return UnaryExp<op::Cos<LhsType>, LhsType>(lhs);
    }

    template<typename LhsType>
    [[nodiscard]] inline UnaryExp<op::Tan<LhsType>, LhsType> tan(const LhsType& lhs) {
        return UnaryExp<op::Tan<LhsType>, LhsType>(lhs);
    }
}// SimpleTensor

#endif //TENSOR_EXP_H