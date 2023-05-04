#ifndef TENSOR_OPER_H
#define TENSOR_OPER_H

#include "shape.h"

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
    }
} // SimpleTensor

#endif //TENSOR_OPER_H
