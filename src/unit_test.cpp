#include <iostream>
#include "tensor.h"
#include "gtest/gtest.h"

TEST(tensorConstructorTest, by_storage_and_shape) {
    st::Tensor A({1, 3, 5, 7, 9, 11, 13, 15}, {2, 2, 2});
    EXPECT_EQ(3, A.n_dim());
    for (st::index_t i = 0; i < A.n_dim(); ++i)
        EXPECT_EQ(2, A.size(i));
    for (st::index_t i = 0; i < 2; ++i)
        for (st::index_t j = 0; j < 2; ++j)
            for (st::index_t k = 0; k < 2; ++k)
                EXPECT_EQ(2*((i*2+j)*2+k+1)-1, (A[{i, j, k}]));
}

TEST(tensorConstructorTest, by_shape) {
    st::Tensor A({2, 2, 2});
    EXPECT_EQ(3, A.n_dim());
    for (st::index_t i = 0; i < A.n_dim(); ++i)
        EXPECT_EQ(2, A.size(i));
    for (st::index_t i = 0; i < 2; ++i)
        for (st::index_t j = 0; j < 2; ++j)
            for (st::index_t k = 0; k < 2; ++k)
                EXPECT_EQ(0, (A[{i, j, k}]));
}

TEST(tensorConstructorTest, copy_constructor) {
    st::Tensor A({1, 3, 5, 7, 9, 11, 13, 15}, {2, 2, 2});
    st::Tensor B = A;
    EXPECT_EQ(3, B.n_dim());
    for (st::index_t i = 0; i < B.n_dim(); ++i)
        EXPECT_EQ(2, B.size(i));
    for (st::index_t i = 0; i < 2; ++i)
        for (st::index_t j = 0; j < 2; ++j)
            for (st::index_t k = 0; k < 2; ++k)
                EXPECT_EQ(2*((i*2+j)*2+k+1)-1, (B[{i, j, k}]));
}

TEST(tensorMakerTest, makers) {
    st::Tensor A = st::Tensor::zeros({2, 2, 2});
    st::Tensor B = st::Tensor::rand({2, 2, 2});
    st::Tensor C = st::Tensor::ones({2, 2, 2});
    EXPECT_EQ(3, A.n_dim());
    for (st::index_t i = 0; i < A.n_dim(); ++i)
        EXPECT_EQ(2, A.size(i));
    for (st::index_t i = 0; i < 2; ++i)
        for (st::index_t j = 0; j < 2; ++j)
            for (st::index_t k = 0; k < 2; ++k) {
                EXPECT_EQ(0, (A[{i, j, k}]));
                EXPECT_EQ(1, (C[{i, j, k}]));
            }
    std::cout << B << std::endl;
}

TEST(tensorCalcOperatorTest, add) {
    st::Tensor A = st::Tensor::rand({2, 2, 2});
    st::Tensor B = st::Tensor::rand({2, 2, 2});
    st::Tensor C = A + B;
    EXPECT_EQ(3, C.n_dim());
    for (st::index_t i = 0; i < C.n_dim(); ++i)
        EXPECT_EQ(2, C.size(i));
    for (st::index_t i = 0; i < 2; ++i)
        for (st::index_t j = 0; j < 2; ++j)
            for (st::index_t k = 0; k < 2; ++k)
                EXPECT_EQ((A[{i, j, k}]+B[{i, j, k}]), (C[{i, j, k}]));
}

TEST(tensorCalcOperatorTest, sub) {
    st::Tensor A = st::Tensor::rand({2, 2, 2});
    st::Tensor B = st::Tensor::rand({2, 2, 2});
    st::Tensor C = A - B;
    EXPECT_EQ(3, C.n_dim());
    for (st::index_t i = 0; i < C.n_dim(); ++i)
        EXPECT_EQ(2, C.size(i));
    for (st::index_t i = 0; i < 2; ++i)
        for (st::index_t j = 0; j < 2; ++j)
            for (st::index_t k = 0; k < 2; ++k)
                EXPECT_EQ((A[{i, j, k}]-B[{i, j, k}]), (C[{i, j, k}]));
}

TEST(tensorCalcOperatorTest, mul) {
    st::Tensor A = st::Tensor::rand({2, 2, 2});
    st::Tensor B = st::Tensor::rand({2, 2, 2});
    st::Tensor C = A * B;
    EXPECT_EQ(3, C.n_dim());
    for (st::index_t i = 0; i < C.n_dim(); ++i)
        EXPECT_EQ(2, C.size(i));
    for (st::index_t i = 0; i < 2; ++i)
        for (st::index_t j = 0; j < 2; ++j)
            for (st::index_t k = 0; k < 2; ++k)
                EXPECT_EQ((A[{i, j, k}]*B[{i, j, k}]), (C[{i, j, k}]));
}

TEST(tensorCalcOperatorTest, div) {
    st::Tensor A = st::Tensor::rand({2, 2, 2});
    st::Tensor B = st::Tensor::rand({2, 2, 2});
    st::Tensor C = A / B;
    EXPECT_EQ(3, C.n_dim());
    for (st::index_t i = 0; i < C.n_dim(); ++i)
        EXPECT_EQ(2, C.size(i));
    for (st::index_t i = 0; i < 2; ++i)
        for (st::index_t j = 0; j < 2; ++j)
            for (st::index_t k = 0; k < 2; ++k)
                EXPECT_EQ((A[{i, j, k}]/B[{i, j, k}]), (C[{i, j, k}]));
}

TEST(tensorCalcOperator, matmul) {
    st::Tensor A = st::Tensor::rand({2, 3});
    st::Tensor B = st::Tensor::rand({3, 4});
    st::Tensor C = st::matmul(A, B);
    EXPECT_EQ(2, C.n_dim());
    EXPECT_EQ(2, C.size(0));
    EXPECT_EQ(4, C.size(1));
    for (st::index_t i = 0; i < 2; ++i)
        for (st::index_t j = 0; j < 4; ++j) {
            st::data_t sum = 0;
            for (st::index_t k = 0; k < 3; ++k)
                sum += A[{i, k}] * B[{k, j}];
            EXPECT_EQ(sum, (C[{i, j}]));
        }
    std::cout << A << std::endl;
    std::cout << B << std::endl;
    std::cout << C << std::endl;
}

TEST(tensorOperatorTest, slice_piece) {
    st::Tensor A = st::Tensor::rand({2, 3, 3});
    st::Tensor B = A.slice(2, 2);
    EXPECT_EQ(3, B.n_dim());
    EXPECT_EQ(2, B.size(0));
    EXPECT_EQ(3, B.size(1));
    EXPECT_EQ(1, B.size(2));
    for (st::index_t i = 0; i < 2; ++i)
        for (st::index_t j = 0; j < 3; ++j)
            EXPECT_EQ((A[{i, j, 2}]), (B[{i, j, 0}]));
    std::cout << A << std::endl;
    std::cout << B << std::endl;
}

TEST(tensorOperatorTest, slice_cube) {
    st::Tensor A = st::Tensor::rand({2, 3, 4});
    st::Tensor B = A.slice(1, 3, 2);
    EXPECT_EQ(3, B.n_dim());
    EXPECT_EQ(2, B.size(0));
    EXPECT_EQ(3, B.size(1));
    EXPECT_EQ(2, B.size(2));
    for (st::index_t i = 0; i < 2; ++i)
        for (st::index_t j = 0; j < 3; ++j)
            for (st::index_t k = 0; k < 2; ++k)
                EXPECT_EQ((A[{i, j, k+1}]), (B[{i, j, k}]));
    std::cout << A << std::endl;
    std::cout << B << std::endl;
}

TEST(tensorOperatorTest, transpose) {
    st::Tensor A = st::Tensor::rand({2, 3, 4});
    st::Tensor B = A.transpose(1, 2);
    EXPECT_EQ(3, B.n_dim());
    EXPECT_EQ(2, B.size(0));
    EXPECT_EQ(4, B.size(1));
    EXPECT_EQ(3, B.size(2));
    for (st::index_t i = 0; i < 2; ++i)
        for (st::index_t j = 0; j < 3; ++j)
            for (st::index_t k = 0; k < 4; ++k)
                EXPECT_EQ((A[{i, j, k}]), (B[{i, k, j}]));
    std::cout << A << std::endl;
    std::cout << B << std::endl;
}
TEST(tensorOperatorTest, reshape) {
    st::Tensor A = st::Tensor::rand({2, 3, 4});
    st::Tensor B = A.view({2, 12});
    EXPECT_EQ(2, B.n_dim());
    EXPECT_EQ(2, B.size(0));
    EXPECT_EQ(12, B.size(1));
    for (st::index_t i = 0; i < 2; ++i)
        for (st::index_t j = 0; j < 12; ++j)
            EXPECT_EQ((A[{i, j / 4, j % 4}]), (B[{i, j}]));
    std::cout << A << std::endl;
    std::cout << B << std::endl;
}

TEST(tensorOperatorTest, permute) {
    st::Tensor A = st::Tensor::rand({2, 3, 4});
    st::Tensor B = A.permute({2, 0, 1});
    EXPECT_EQ(3, B.n_dim());
    EXPECT_EQ(4, B.size(0));
    EXPECT_EQ(2, B.size(1));
    EXPECT_EQ(3, B.size(2));
    for (st::index_t i = 0; i < 2; ++i)
        for (st::index_t j = 0; j < 3; ++j)
            for (st::index_t k = 0; k < 4; ++k)
                EXPECT_EQ((A[{i, j, k}]), (B[{k, i, j}]));
    std::cout << A << std::endl;
    std::cout << B << std::endl;
}
TEST(tensorOperatorTest, sumInOneDim) {
    st::Tensor A = st::Tensor::rand({2, 3, 4});
    st::Tensor B = A.sum(1);
    EXPECT_EQ(2, B.n_dim());
    EXPECT_EQ(2, B.size(0));
    EXPECT_EQ(4, B.size(1));
    for (st::index_t i = 0; i < 2; ++i)
        for (st::index_t k = 0; k < 4; ++k) {
            st::data_t sum = 0;
            for (st::index_t j = 0; j < 3; ++j)
                sum += A[{i, j, k}];
            EXPECT_EQ(sum, (B[{i, k}]));
        }
    std::cout << A << std::endl;
    std::cout << B << std::endl;
}

TEST(tensorBroadcastTest, broadcast) {
    st::Tensor A = st::Tensor::rand({2, 3, 4});
    st::Tensor B = st::Tensor::rand({3, 4});
    st::Tensor C = A + B;
    EXPECT_EQ(3, C.n_dim());
    EXPECT_EQ(2, C.size(0));
    EXPECT_EQ(3, C.size(1));
    EXPECT_EQ(4, C.size(2));
    for (st::index_t i = 0; i < 2; ++i)
        for (st::index_t j = 0; j < 3; ++j)
            for (st::index_t k = 0; k < 4; ++k)
                EXPECT_EQ((A[{i, j, k}] + B[{j, k}]), (C[{i, j, k}]));
    std::cout << A << std::endl;
    std::cout << B << std::endl;
    std::cout << C << std::endl;
}

TEST(tensorIteratorTest, iterator) {
    st::Tensor A = st::Tensor::rand({2, 3, 4});
    st::Tensor::iterator it = A.begin();
    for (st::index_t i = 0; i < 2; ++i)
        for (st::index_t j = 0; j < 3; ++j)
            for (st::index_t k = 0; k < 4; ++k) {
                EXPECT_EQ((A[{i, j, k}]), (*it));
                ++it;
            }
    EXPECT_EQ(it, A.end());
}
TEST(tensorIteratorTest, constIterator) {
    const st::Tensor A = st::Tensor::rand({2, 3, 4});
    st::Tensor::const_iterator it = A.begin();
    for (st::index_t i = 0; i < 2; ++i)
        for (st::index_t j = 0; j < 3; ++j)
            for (st::index_t k = 0; k < 4; ++k) {
                EXPECT_EQ((A[{i, j, k}]), (*it));
                ++it;
            }
    EXPECT_EQ(it, A.end());
}


TEST(tensorExpLazyCaculationTest, lazyEvaluation) {
    st::Tensor A = st::Tensor::rand({2, 3});
    st::Tensor B = st::Tensor::rand({2, 3});
    auto C = A + B * A;
    st::Tensor res = C;
    EXPECT_EQ(2, res.n_dim());
    EXPECT_EQ(2, res.size(0));
    EXPECT_EQ(3, res.size(1));
    for (st::index_t i = 0; i < 2; ++i)
        for (st::index_t j = 0; j < 3; ++j)
            EXPECT_EQ((A[{i, j}] + B[{i, j}] * A[{i, j}]), (res[{i, j}]));
    std::cout << "before change:" << std::endl;
    std::cout << A << std::endl;
    std::cout << B << std::endl;
    std::cout << res << std::endl;
    A[{0, 0}] = 0; A[{0, 1}] = 1; A[{0, 2}] = 2;
    A[{1, 0}] = 3; A[{1, 1}] = 4; A[{1, 2}] = 5;
    res = C;
    for (st::index_t i = 0; i < 2; ++i)
        for (st::index_t j = 0; j < 3; ++j)
            EXPECT_EQ((A[{i, j}] + B[{i, j}] * A[{i, j}]), (res[{i, j}]));
    std::cout << "after change:" << std::endl;
    std::cout << A << std::endl;
    std::cout << B << std::endl;
    std::cout << res << std::endl;
}

TEST(tensorErrorCheck, outOfRange) {
    st::Tensor A = st::Tensor::rand({2, 3});
    EXPECT_THROW((A[{2, 0}]), st::err::Error);
    EXPECT_THROW((A[{0, 3}]), st::err::Error);
    EXPECT_THROW((A[{2, 3}]), st::err::Error);
    EXPECT_THROW((A[{0, 0, 0}]), st::err::Error);
    EXPECT_THROW((A.size(10)), st::err::Error);
    EXPECT_THROW((A.transpose(12, 35)), st::err::Error);
    EXPECT_THROW((A.slice(10, 3)), st::err::Error);
    EXPECT_THROW((A.slice(0, 3, 14)), st::err::Error);
    EXPECT_THROW((A.slice(10, 4, 1)), st::err::Error);
}

TEST(tensorErrorCheck, calculationFailed) {
    st::Tensor A = st::Tensor::rand({2, 8});
    st::Tensor B = st::Tensor::rand({3, 4});
    EXPECT_THROW(({ st::Tensor res = A + B; }), st::err::Error);
    EXPECT_THROW(({ st::Tensor res = A - B; }), st::err::Error);
    EXPECT_THROW(({ st::Tensor res = A * B; }), st::err::Error);
    EXPECT_THROW(({ st::Tensor res = A / B; }), st::err::Error);
    EXPECT_THROW(({ st::Tensor res = st::matmul(A, B); }), st::err::Error);
}

TEST(tensorApplicationTest, linearRegression) {
    const int batch_size = 5;
    const int dim = 2;
    st::Tensor W({1, dim});
    st::Tensor B({1, 1});
    st::Tensor X({
        1, 2,
        4, 5,
        6, 7,
        8, 9,
        10, 11
        }, {batch_size, dim, 1});
    st::Tensor t_Y({
        3,
        9,
        13,
        17,
        21
        }, {batch_size, 1});
    t_Y = t_Y.view({batch_size, 1, 1});
    double learning_rate = 0.00001;
    W = st::Tensor::rand({1, dim});
    st::Tensor loss({10, 1, 1,});
    for (int i = 0; i < 10000; ++i) {
        auto Y = matmul(W, X) + B;
        auto dY = Y - t_Y;
        auto L = dY * dY;
        loss = L;
        st::Tensor dy = dY;
        st::Tensor dw = matmul(X, dY);
        W = W - learning_rate / batch_size * 2 * dw.sum(0).transpose(0, 1);
        B = B - learning_rate / batch_size * 2 * dy.sum(0).transpose(0, 1);
    }
    std::cout << "W:" << std::endl;
    std::cout << W << std::endl;
    std::cout << "B:" << std::endl;
    std::cout << B << std::endl;
    std::cout << "loss:" << std::endl;
    std::cout << loss.sum() << std::endl;
}