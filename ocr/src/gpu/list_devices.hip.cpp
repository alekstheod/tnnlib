#include <hip/hip_runtime.h>
#include <cstdio>

int main() {
    int n;
    hipGetDeviceCount(&n);
    printf("HIP_VISIBLE_DEVICES=%s\n", getenv("HIP_VISIBLE_DEVICES") ?: "(unset)");
    for (int i = 0; i < n; ++i) {
        hipDeviceProp_t p;
        hipGetDeviceProperties(&p, i);
        printf("Device %d: %s\n", i, p.name);
    }
    return 0;
}
