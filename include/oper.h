	#ifndef TENSOR_OPER_H
#define TENSOR_OPER_H

#include "shape.h"
#include <cmath>

namespace st {
    namespace op {
        template<typename LhsType, typename RhsType>
        struct Add {
            static double eval(IndexArray ids, LhsType lhs, RhsType rhs) {
                return lhs.eval(ids)+rhs.eval(ids);
            }
        };
        template<typename LhsType, typename RhsType>
        struct Sub {
            static double eval(IndexArray ids, LhsType lhs, RhsType rhs) {
                return lhs.eval(ids)-rhs.eval(ids);
            }
        };
        template<typename LhsType, typename RhsType>
        struct Mul {
            static double eval(IndexArray ids, LhsType lhs, RhsType rhs) {
                return lhs.eval(ids)*rhs.eval(ids);
            }
        };
        template<typename LhsType, typename RhsType>
        struct Div {
            static double eval(IndexArray ids, LhsType lhs, RhsType rhs) {
                return lhs.eval(ids)/rhs.eval(ids);
            }
        };
        template<typename LhsType>
        struct Neg {
            static double eval(IndexArray ids, LhsType lhs) {
                return -lhs.eval(ids);
            }
        };
        template<typename LhsType>
        struct Sin {
            static double eval(IndexArray ids, LhsType lhs) {
                return std::sin(lhs.eval(ids));
            }
        };
        template<typename LhsType>
        struct Cos {
            static double eval(IndexArray ids, LhsType lhs) {
                return std::cos(lhs.eval(ids));
            }
        };
        template<typename LhsType>
        struct Tan {
            static double eval(IndexArray ids, LhsType lhs) {
                return std::tan(lhs.eval(ids));
            }
        };
    }
} // SimpleTensor

#endif //TENSOR_OPER_H
