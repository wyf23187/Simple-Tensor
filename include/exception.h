#ifndef TENSOR__INCLUDE_EXCEPTION_H_
#define TENSOR__INCLUDE_EXCEPTION_H_
#include "allocator.h"
#include <exception>
#include <algorithm>
namespace st {
	namespace err
	{
		struct Error: public std::exception {
			Error(const char* file, const char* func, unsigned int line);
			const char* what() const noexcept;

			static char msg_[300];
			const char* file_;
			const char* func_;
			const unsigned int line_;
		};
	}
	#define ERROR_LOCATION __FILE__, __func__, __LINE__
	#define THROW_ERROR(format, ...)	do {	\
    std::sprintf(::st::err::Error::msg_, (format), ##__VA_ARGS__);    \
    throw ::st::err::Error(ERROR_LOCATION);                           \
	} while(0)
	#ifndef CANCEL_CHECK
	// base assert macro
	#define CHECK_TRUE(expr, format, ...) \
		if(!(expr)) THROW_ERROR((format), ##__VA_ARGS__)

	#define CHECK_NOT_NULL(ptr, format, ...) \
		if(nullptr == (ptr)) THROW_ERROR((format), ##__VA_ARGS__)

	#define CHECK_EQUAL(x, y, format, ...) \
		if((x) != (y)) THROW_ERROR((format), ##__VA_ARGS__)

	#define CHECK_IN_RANGE(x, lower, upper, format, ...) \
		if((x) < (lower) || (x) >= (upper)) THROW_ERROR((format), ##__VA_ARGS__)

	#define CHECK_FLOAT_EQUAL(x, y, format, ...) \
		if(std::fabs((x)-(y)) < 1e-4) THROW_ERROR((format), ##__VA_ARGS__)

	#define CHECK_INDEX_VALID(x, format, ...) \
		if((x) > INDEX_MAX) THROW_ERROR((format), ##__VA_ARGS__)
	#define CHECK_EXP_SAME_SHAPE(e1_, e2_) do { \
    auto& e1 = (e1_);  \
    auto& e2 = (e2_);  \
    CHECK_EQUAL(e1.ndim(), e2.ndim(),  \
        "Expect the same dimensions, but got %dD and %dD",  \
        e1.ndim(), e2.ndim());  \
    for(index_t i = 0; i < e1.ndim(); ++i) \
        CHECK_EQUAL(e1.size(i), e2.size(i),  \
            "Expect the same size on the %d dimension, but got %d and %d.",  \
            i, e1.size(i), e2.size(i));  \
	} while(0)
    #define CHECK_EXP_BROADCAST(e1_, e2_) do { \
    auto& e1 = (e1_);                          \
    auto& e2 = (e2_);                          \
    int i = e1->n_dim()-1;                   \
    int j = e2->n_dim()-1;                   \
    for (; i >= 0 && j >= 0; --i, --j) {   \
        CHECK_TRUE(e1->size(i) == e2->size(j) || e1->size(i) == 1 || e2->size(j) == 1, \
            "Broadcast error with %d in tensor a but %d in tensor b.", e1->size(i), e2->size(j) \
        );                                     \
    }                                      \
    } while(0);
	#else
	#define CHECK_TRUE(expr, format, ...) {}
	#define CHECK_NOT_NULL(ptr, format, ...) {}
	#define CHECK_EQUAL(x, y, format, ...) {}
	#define CHECK_IN_RANGE(x, lower, upper, format, ...) {}
	#define CHECK_FLOAT_EQUAL(x, y, format, ...) {}
	#define CHECK_INDEX_VALID(x, format, ...) {}
	#define CHECK_EXP_SAME_SHAPE(e1, e2) {}
	#define CHECK_EXP_BROADCAST(e1, e2) {}

	#endif

} // SimpleTensor
#endif //TENSOR__INCLUDE_EXCEPTION_H_
