#ifndef TENSOR_OPER_H
#define TENSOR_OPER_H

#include "shape.h"
#include "exp.h"
#include <cmath>

namespace st {
    namespace op {
        struct Add {
            static data_t eval(data_t lhs, data_t rhs) {
                return lhs+rhs;
            }
        };
        struct Sub {
            static data_t eval(data_t lhs, data_t rhs) {
                return lhs-rhs;
            }
        };
        struct Mul {
            static data_t eval(data_t lhs, data_t rhs) {
                return lhs*rhs;
            }
        };
        struct Div {
            static data_t eval(data_t lhs, data_t rhs) {
                return lhs/rhs;
            }
        };
        struct Neg {
            static data_t eval(data_t lhs) {
                return -lhs;
            }
        };
        struct Sin {
            static data_t eval(data_t lhs) {
                return std::sin(lhs);
            }
        };
        struct Cos {
            static data_t eval(data_t lhs) {
                return std::cos(lhs);
            }
        };
        struct Tan {
            static data_t eval(data_t lhs) {
                return std::tan(lhs);
            }
        };
    } // op

    template<typename LhsType, typename RhsType>
    [[nodiscard]] inline Exp<BinaryExp<op::Add, LhsType, RhsType>> operator+(const Exp<LhsType>& lhs, const Exp<RhsType>& rhs) {
        return Exp<BinaryExp<op::Add, LhsType, RhsType>>(
                std::make_shared<BinaryExp<op::Add, LhsType, RhsType>>(lhs.ptr(), rhs.ptr())
        );
    }

    template<typename LhsType, typename RhsType>
    [[nodiscard]] inline Exp<BinaryExp<op::Sub, LhsType, RhsType>> operator-(const Exp<LhsType>& lhs, const Exp<RhsType>& rhs) {
        return Exp<BinaryExp<op::Sub, LhsType, RhsType>>(
                std::make_shared<BinaryExp<op::Sub, LhsType, RhsType>>(lhs.ptr(), rhs.ptr())
        );
    }

    template<typename LhsType, typename RhsType>
    [[nodiscard]] inline Exp<BinaryExp<op::Mul, LhsType, RhsType>> operator*(const Exp<LhsType>& lhs, const Exp<RhsType>& rhs) {
        return Exp<BinaryExp<op::Mul, LhsType, RhsType>>(
                std::make_shared<BinaryExp<op::Mul, LhsType, RhsType>>(lhs.ptr(), rhs.ptr())
        );
    }

    template<typename LhsType, typename RhsType>
    [[nodiscard]] inline Exp<BinaryExp<op::Div, LhsType, RhsType>> operator/(const Exp<LhsType>& lhs, const Exp<RhsType>& rhs) {
        return Exp<BinaryExp<op::Div, LhsType, RhsType>>(
                std::make_shared<BinaryExp<op::Div, LhsType, RhsType>>(lhs.ptr(), rhs.ptr())
        );
    }

    template<typename LhsType>
    [[nodiscard]] inline UnaryExp<op::Neg, LhsType> operator-(const Exp<LhsType>& lhs) {
        return UnaryExp<op::Neg, LhsType>(lhs.ptr());
    }

    template<typename LhsType>
    [[nodiscard]] inline UnaryExp<op::Sin, LhsType> sin(const Exp<LhsType>& lhs) {
        return UnaryExp<op::Sin, LhsType>(lhs.ptr());
    }

    template<typename LhsType>
    [[nodiscard]] inline UnaryExp<op::Cos, LhsType> cos(const Exp<LhsType>& lhs) {
        return UnaryExp<op::Cos, LhsType>(lhs.ptr());
    }

    template<typename LhsType>
    [[nodiscard]] inline UnaryExp<op::Tan, LhsType> tan(const Exp<LhsType>& lhs) {
        return UnaryExp<op::Tan, LhsType>(lhs.ptr());
    }
} // st

#endif //TENSOR_OPER_H