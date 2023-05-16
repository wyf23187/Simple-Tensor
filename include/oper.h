#ifndef TENSOR_OPER_H
#define TENSOR_OPER_H

#include "shape.h"
#include "exp.h"
#include <cmath>

namespace st {
    namespace op {
        struct Add {
            template<typename LhsType, typename RhsType>
            static data_t eval(IndexArray& idx, std::shared_ptr<LhsType> lhs, std::shared_ptr<RhsType> rhs) {
                return lhs->eval(idx)+rhs->eval(idx);
            }
            template<typename LhsType, typename RhsType>
            static const Shape& size(const std::shared_ptr<LhsType>& lhs, const std::shared_ptr<RhsType>& rhs) {
                return lhs->size();
            }
            template<typename LhsType, typename RhsType>
            static index_t size(index_t idx, const std::shared_ptr<LhsType>& lhs, const std::shared_ptr<RhsType>& rhs) {
                if (idx >= lhs->ndim()) return rhs->size(idx);
                if (idx >= rhs->ndim()) return lhs->size(idx);
                return max(lhs->size(idx), rhs->size(idx));
            }
            template<typename LhsType, typename RhsType>
            static index_t ndim(const std::shared_ptr<LhsType>& lhs, const std::shared_ptr<RhsType>& rhs) {
                return max(lhs->ndim(), rhs->ndim());
            }
        };
        struct Sub {
            template<typename LhsType, typename RhsType>
            static data_t eval(IndexArray& idx, std::shared_ptr<LhsType> lhs, std::shared_ptr<RhsType> rhs) {
                return lhs->eval(idx)-rhs->eval(idx);
            }
            template<typename LhsType, typename RhsType>
            static const Shape& size(const std::shared_ptr<LhsType>& lhs, const std::shared_ptr<RhsType>& rhs) {
                return lhs->size();
            }
        };
        struct Mul {
            template<typename LhsType, typename RhsType>
            static data_t eval(IndexArray& idx, std::shared_ptr<LhsType> lhs, std::shared_ptr<RhsType> rhs) {
                return lhs->eval(idx)*rhs->eval(idx);
            }
            template<typename LhsType, typename RhsType>
            static const Shape& size(const std::shared_ptr<LhsType>& lhs, const std::shared_ptr<RhsType>& rhs) {
                return lhs->size();
            }
        };
        struct Div {
            template<typename LhsType, typename RhsType>
            static data_t eval(IndexArray& idx, std::shared_ptr<LhsType> lhs, std::shared_ptr<RhsType> rhs) {
                return lhs->eval(idx)/rhs->eval(idx);
            }
            template<typename LhsType, typename RhsType>
            static const Shape& size(const std::shared_ptr<LhsType>& lhs, const std::shared_ptr<RhsType>& rhs) {
                return lhs->size();
            }
        };
        struct MatrixMul_2dim {
            template<typename LhsType, typename RhsType>
            static data_t eval(IndexArray& idx, std::shared_ptr<LhsType> lhs, std::shared_ptr<RhsType> rhs) {
                const Shape& ls = lhs->size();
                const Shape& rs = rhs->size();
                index_t l0 = ls[0], l1 = ls[1], r0 = rs[0], r1 = rs[1];
                // default l1 == r0
                // default lhs and rhs is 2-dimensional
                data_t res = 0;
                for (index_t i = 0; i < l1; ++i) {
                    res += lhs->eval({idx[0], i})*rhs->eval({i, idx[1]});
                }
                return res;
            }
            template<typename LhsType, typename RhsType>
            static Shape size(const std::shared_ptr<LhsType>& lhs, const std::shared_ptr<RhsType>& rhs) {
                return Shape({lhs->size()[0], rhs->size()[1]});
            }
        };
        struct MatrixMul_3dim {
            template<typename LhsType, typename RhsType>
            static data_t eval(IndexArray& idx, std::shared_ptr<LhsType> lhs, std::shared_ptr<RhsType> rhs) {
                const Shape& ls = lhs->size();
                const Shape& rs = rhs->size();
                index_t l0 = ls[0], l1 = ls[1], l2 = ls[2], r0 = rs[0], r1 = rs[1], r2 = rs[2];
                // default l2 == r1
                // default lhs and rhs is 3-dimensional
                data_t res = 0;
                for (index_t i = 0; i < l2; ++i) {
                    res += lhs->eval({idx[0], idx[1], i})*rhs->eval({idx[0], i, idx[2]});
                }
                return res;
            }
            template<typename LhsType, typename RhsType>
            static Shape size(const std::shared_ptr<LhsType>& lhs, const std::shared_ptr<RhsType>& rhs) {
                return Shape({lhs->size()[0], lhs->size()[1], rhs->size()[2]});
            }
        };
        struct MatrixMul {
            template<typename LhsType, typename RhsType>
            static data_t eval(IndexArray& idx, std::shared_ptr<LhsType> lhs, std::shared_ptr<RhsType> rhs) {
                int l0, l1;
                l0 = lhs->size()[lhs->n_dim()-2];
                l1 = lhs->size()[lhs->n_dim()-1];
                int r0, r1;
                r0 = rhs->size()[rhs->n_dim()-2];
                r1 = rhs->size()[rhs->n_dim()-1];
                data_t res = 0;
                for (int i = 0; i < l0; ++i) {
                    IndexArray lidx = idx;
                    IndexArray ridx = idx;
                    lidx[lhs->n_dim()-1] = i;
                    ridx[rhs->n_dim()-2] = i;
                    res += lhs->eval(lidx)*rhs->eval(ridx);
                }
                return res;
            }
            template<typename LhsType, typename RhsType>
            static Shape size(const std::shared_ptr<LhsType>& lhs, const std::shared_ptr<RhsType>& rhs) {
                Shape res(std::max(lhs->n_dim(), rhs->n_dim()));
                int n = res.n_dim();
                int nl = lhs->n_dim()-2, nr = rhs->n_dim()-2;
                for (int i = 0; i < n-2; ++i) {
                    if (n-2-nl > i) res[i] = rhs->size()[n-2-nr+i];
                    else if (n-2-nr > i) res[i] = lhs->size()[n-2-nl+i];
                    else res[i] = std::max(lhs->size()[i-(n-2-nl)], rhs->size()[i-(n-2-nr)]);
                }
                res[n-2] = lhs->size()[lhs->n_dim()-2];
                res[n-1] = rhs->size()[rhs->n_dim()-1];
                return res;
            }
        };
        struct Neg {
            template<typename LhsType>
            static data_t eval(IndexArray& idx, std::shared_ptr<LhsType> lhs) {
                return -lhs->eval(idx);
            }
            template<typename LhsType, typename RhsType>
            static const Shape& size(const std::shared_ptr<LhsType>& lhs, const std::shared_ptr<RhsType>& rhs) {
                return lhs->size();
            }
        };
        struct Sin {
            template<typename LhsType>
            static data_t eval(IndexArray& idx, std::shared_ptr<LhsType> lhs) {
                return std::sin(lhs->eval(idx));
            }
            template<typename LhsType, typename RhsType>
            static const Shape& size(const std::shared_ptr<LhsType>& lhs, const std::shared_ptr<RhsType>& rhs) {
                return lhs->size();
            }
        };
        struct Cos {
            template<typename LhsType>
            static data_t eval(IndexArray& idx, std::shared_ptr<LhsType> lhs) {
                return std::cos(lhs->eval(idx));
            }
            template<typename LhsType, typename RhsType>
            static const Shape& size(const std::shared_ptr<LhsType>& lhs, const std::shared_ptr<RhsType>& rhs) {
                return lhs->size();
            }
        };
        struct Tan {
            template<typename LhsType>
            static data_t eval(IndexArray& idx, std::shared_ptr<LhsType> lhs) {
                return std::tan(lhs->eval(idx));
            }
            template<typename LhsType, typename RhsType>
            static const Shape& size(const std::shared_ptr<LhsType>& lhs, const std::shared_ptr<RhsType>& rhs) {
                return lhs->size();
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

    template<typename LhsType, typename RhsType>
    [[nodiscard]] inline Exp<BinaryExp<op::MatrixMul_2dim, LhsType, RhsType>> mm(const Exp<LhsType>& lhs, const Exp<RhsType>& rhs) {
        return Exp<BinaryExp<op::MatrixMul_2dim, LhsType, RhsType>>(
                std::make_shared<BinaryExp<op::MatrixMul_2dim, LhsType, RhsType>>(lhs.ptr(), rhs.ptr())
        );
    }

    template<typename LhsType, typename RhsType>
    [[nodiscard]] inline Exp<BinaryExp<op::MatrixMul_3dim, LhsType, RhsType>> bmm(const Exp<LhsType>& lhs, const Exp<RhsType>& rhs) {
        return Exp<BinaryExp<op::MatrixMul_3dim, LhsType, RhsType>>(
                std::make_shared<BinaryExp<op::MatrixMul_3dim, LhsType, RhsType>>(lhs.ptr(), rhs.ptr())
        );
    }

    template<typename LhsType, typename RhsType>
    [[nodiscard]] inline Exp<BinaryExp<op::MatrixMul, LhsType, RhsType>> matmul(const Exp<LhsType>& lhs, const Exp<RhsType>& rhs) {
        return Exp<BinaryExp<op::MatrixMul, LhsType, RhsType>>(
                std::make_shared<BinaryExp<op::MatrixMul, LhsType, RhsType>>(lhs.ptr(), rhs.ptr())
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