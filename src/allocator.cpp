#include "allocator.h"
#include <memory>
#include <cstdlib>

namespace st {
    index_t Alloc::allocate_memory_size = 0;
    index_t Alloc::deallocate_memory_size = 0;
    Alloc& Alloc::self() {
        static Alloc alloc;
        return alloc;
    }

    void* Alloc::allocate(index_t size) {
        auto iter = self().cache_.find(size);
        void* res;
        if (iter != self().cache_.end()) {
            res = iter->second.release();
            self().cache_.erase(iter);
        } else {
            res = std::malloc(size);
            if (res == nullptr) {
                puts("No Enough memory!");
            }
        }
        allocate_memory_size += size;
        return res;
    }

    void Alloc::deallocate(void* ptr, index_t size) {
        deallocate_memory_size -= size;
        self().cache_.emplace(size, ptr);
    }

    bool Alloc::all_clear() {
        return deallocate_memory_size == allocate_memory_size;
    }
}