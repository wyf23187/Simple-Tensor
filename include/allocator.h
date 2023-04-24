#ifndef TENSOR_ALLOCATOR_H
#define TENSOR_ALLOCATOR_H

// basic allocate

#include <cstdlib>
#include <map>
#include <memory>
#include <iostream>

namespace SimpleTensor {
    typedef unsigned int index_t;
    class Alloc {
    public:
        class trivial_delete_handler {
        public:
            explicit trivial_delete_handler(index_t size_): size(size_) {}
            void operator()(void *ptr) { deallocate(ptr, size); }
        private:
            index_t size;
        };

        template<typename T>
        class nontrivial_delete_handler {
        public:
            void operator()(void *ptr) {
                static_cast<T*>(ptr)->~T();
                deallocate(ptr, sizeof(T));
            }
        };

        template<typename T>
        using TrivalUniquePtr = std::unique_ptr<T, trivial_delete_handler>;

        template<typename T>
        using NonTrivalUniquePtr = std::unique_ptr<T, nontrivial_delete_handler<T>>;

        // make deleter
        template<typename T>
        static std::shared_ptr<T> shared_allocate(index_t n_bytes) {
            void *raw_ptr = allocate(n_bytes);
            return std::shared_ptr<T>(static_cast<T*>(raw_ptr), trivial_delete_handler(n_bytes));
        }

        template<typename T>
        static TrivalUniquePtr<T> unique_allocate(index_t n_bytes) {
            void *raw_ptr = allocate(n_bytes);
            return TrivalUniquePtr<T>(static_cast<T*>(raw_ptr), trivial_delete_handler(n_bytes));
        }

        // make constructor
        template<typename T, typename... Args>
        static std::shared_ptr<T> shared_construct(Args&&...args) {
            void *raw_ptr = allocate(sizeof(T));
            new(raw_ptr) T(std::forward<Args>(args)...);
            return std::shared_ptr<T>(static_cast<T*>(raw_ptr), nontrivial_delete_handler<T>());
        }

        template<typename T, typename... Args>
        static NonTrivalUniquePtr<T> unique_construct(Args&&...args) {
            void *raw_ptr = allocate(sizeof(T));
            new(raw_ptr) T(std::forward<Args>(args)...);
            return NonTrivalUniquePtr<T>(static_cast<T*>(raw_ptr), nontrivial_delete_handler<T>());
        }
        static bool all_clear();

    private:
        Alloc() = default;
        ~Alloc() {
            for (auto & iter : cache_)
                iter.second.release();
        }
        static Alloc &self();

        static void* allocate(index_t size);
        static void deallocate(void* ptr, index_t size);

        static index_t allocate_memory_size;
        static index_t deallocate_memory_size;

        struct free_deleter {
            void operator()(void *ptr) { std::free(ptr); }
        };
        std::multimap<index_t, std::unique_ptr<void, free_deleter>> cache_;
    };
} // SimpleTensor

#endif //TENSOR_ALLOCATOR_H
