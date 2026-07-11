#include <stdio.h>

__global__ void helloFromGpu(void) {
    printf("on gpu\n");
    printf("threadIdx: %d %d %d\n", threadIdx.x, threadIdx.y, threadIdx.z);
    printf("blockIdx: %d %d %d\n", blockIdx.x, blockIdx.y, blockIdx.z);
    printf("blockDim: %d %d %d\n", blockDim.x, blockDim.y, blockDim.z);
    printf("gridDim: %d %d %d\n", gridDim.x, gridDim.y, gridDim.z);
}

void helloFromCpu(void) {
    printf("on cpu\n");
}

int main() {
    
    helloFromGpu<<<3,3>>>(); // <<<grid dimension, block dimension>>> aka number of blocks , number of threads per block

    helloFromCpu();

    // cleans up and destroys resources on current device process
    cudaDeviceReset();

    return 0;
}