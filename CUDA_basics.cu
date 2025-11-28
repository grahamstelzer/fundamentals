// todo: blow up the gpu with the vec_add function

#include <iostream> // nvcc compiles c++ by default
#include <stdio.h> // early examples use c for faster compile/simpler syntax

#include <cuda_runtime.h>


// hello world example
__global__ void hello_kernel() {
    printf("on thread %d\n", threadIdx.x);
}

int hello() {
    printf("1 block, 4 threads:\n");
    hello_kernel<<<1, 4>>>();
    cudaDeviceSynchronize(); // wait for GPU to finish tasks
    // note: dev sync necessary to finish first hello_kernel() before second

    printf("\n2 blocks of 4 threads:\n");
    hello_kernel<<<2, 4>>>();
    cudaDeviceSynchronize(); 
    return 0;
}


// vector addition (this is like the hello world for cuda)
__global__ void vec_add_kernel(const float* v1_ptr, const float* v2_ptr, float* v3_ptr, int v_len) {
    printf("blockIdx.x: %d, blockDim.x: %d, threadIdx.x: %d\n", blockIdx.x, blockDim.x, threadIdx.x);
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // add into non-const vector
    if (i < v_len) {
        printf("computing %f + %f\n", v1_ptr[i], v2_ptr[i]);
        v3_ptr[i] = v1_ptr[i] + v2_ptr[i];
    }
}

int vec_add() {
    // methodology: must create on host, send to gpu memory for kernel, send back

    int n = 16; // num elements
    size_t size = n * sizeof(float); // mem size

    // create/populate host vectors (h_ prefix for host)
    float h_a[n], h_b[n], h_c[n];
    for (int i = 0; i < n; i++) {
        h_a[i] = float(i);
        h_b[i] = 1.0; // float(i * 2) // just add 1 for this go
    }

    // create pointers for device vectors, alloc with cudaMalloc
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // send to gpu
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // run kernel
    int threads = 16;
    int blocks = 1;
    vec_add_kernel<<<blocks, threads>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    // vec_add_kernel calculation occurs

    // copy back
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    printf("vec_add_kernel results (placed into c vector)\n");
    for (int i = 0; i < n; i++) {
        printf("%f ", h_c[i]);
    }
    printf("\n");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}


// intro to shared memory (C++)

// this function, when all instances of threads run, loads all elements 
//  from a global memory location into shared memory
//  then multiplies them by 2.0 back to global 
__global__ void shared_mem_kernel(const float* in, float* out) {
    extern __shared__ float tile[]; // see notes but extern = dynamic, __shared__ = memory location

    int tid = threadIdx.x; // remember each thread has unique id
    tile[tid] = in[tid];

    // ensure all threads in block are finished before reading to out
    __syncthreads();

    out[tid] = tile[tid] * 2.0f;
}

int shared_mem() {
    // setup host mem array
    int n = 16;
    size_t size = n * sizeof(float);
    float h_in[n], h_out[n];
    // populate with floats
    for(int i = 0; i < n; i++) {
        h_in[i] = float(i);
    }

    // setup device arrays as pointers
    float *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    // send to device
    // remember: cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind)
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    // call function
    shared_mem_kernel<<<1, n, size>>>(d_in, d_out);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    std::cout << "\nshared mem output:\n";
    for(int i = 0; i < n; i++) {
        std::cout << h_out[i] << " ";
    }
    std::cout << std::endl;

    return 0;

}


// warp-level reuduction
//  notes very helpful for syntax and visualization in this example
//  methodology: each thread is given a value, then a tree reduction occurs
//      this means that shared memory is never used, we just look at registers

__inline__ __device__ float warp_reduce_sum(float val) {
    // methodology:
    //  magic is __shfl_down_sync() lets threads read register [curr + offset] in the same warp
    //      this value is placed into val and summated
    //  0xffffffff is a mask, all threads "1" and therefore are used
    //  looping halves the offset each time
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// one-block example (though it is setup for multiple blocks)
__global__ void warp_reduce_kernel(const float* x, float* out) {
    float val = x[threadIdx.x];
    float sum = warp_reduce_sum(val);

    // check for threads at every warp step (32)
    // write the values of these threads into out
    if ((threadIdx.x & 31) == 0) {
        out[threadIdx.x / 32] = sum;
    }
}

int warp_reduce() {

    // setup host vars
    const int n = 64;               // 64 elements → 2 warps
    size_t size = n * sizeof(float);

    float h_x[n];
    for (int i = 0; i < n; i++) {
        h_x[i] = 1.0f;
    }

    // setup device ptrs
    float *d_x, *d_out;
    int num_warps = (n + 31) / 32;  // for d_out we only need space for few results
    cudaMalloc(&d_x, size);
    cudaMalloc(&d_out, num_warps * sizeof(float));

    // send input to device
    cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);

    // launch kernel
    warp_reduce_kernel<<<1, n>>>(d_x, d_out);
    // sync
    cudaDeviceSynchronize();

    // get results
    float h_out[num_warps];
    cudaMemcpy(h_out, d_out, num_warps * sizeof(float), cudaMemcpyDeviceToHost);


    // debug prints:

    // per warp
    std::cout << "Warp sums: ";
    for (int i = 0; i < num_warps; i++) {
        std::cout << h_out[i] << " ";
    }
    std::cout << std::endl;

    // total
    float total = 0.0f;
    for (int i = 0; i < num_warps; i++) total += h_out[i];
    std::cout << "Total sum: " << total << std::endl;
    
    return 0;
}



int main() {

    hello();
    vec_add();
    shared_mem();
    warp_reduce();

    return 0;
}





















// below is like a bit more advanced, will slowly add

// #define CHECK(err) if (err != cudaSuccess){ \
//     std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; exit(1); }


// // ---------------------------------------------------------------
// // 3. WARP SHUFFLE EXAMPLE (warp-level reduction)
// // ---------------------------------------------------------------

// __inline__ __device__ float warp_reduce_sum(float val) {
//     for (int offset = 16; offset > 0; offset /= 2) {
//         val += __shfl_down_sync(0xffffffff, val, offset);
//     }
//     return val;
// }

// // one-block example
// __global__ void warp_reduce_kernel(const float* x, float* out) {
//     float val = x[threadIdx.x];
//     float sum = warp_reduce_sum(val);

//     // write once per warp
//     if ((threadIdx.x & 31) == 0)
//         out[threadIdx.x / 32] = sum;
// }

// // ---------------------------------------------------------------
// // 4. BLOCK REDUCTION (shared memory + warp shuffle)
// // ---------------------------------------------------------------

// __global__ void block_reduce_kernel(const float* x, float* out) {
//     extern __shared__ float smem[];
//     int tid = threadIdx.x;

//     // load
//     smem[tid] = x[tid];
//     __syncthreads();

//     // reduce in shared memory
//     for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
//         if (tid < stride) smem[tid] += smem[tid + stride];
//         __syncthreads();
//     }

//     // final warp reduce
//     float v = smem[tid];
//     if (tid < 32) v = warp_reduce_sum(v);

//     if (tid == 0) out[0] = v;
// }

// // ---------------------------------------------------------------
// // 5. TILED MATRIX MULTIPLY
// // ---------------------------------------------------------------

// #define TILE 16

// __global__ void matmul_tiled(const float* A, const float* B, float* C, int N) {
//     __shared__ float As[TILE][TILE];
//     __shared__ float Bs[TILE][TILE];

//     int row = blockIdx.y * TILE + threadIdx.y;
//     int col = blockIdx.x * TILE + threadIdx.x;

//     float sum = 0.0f;

//     for (int t = 0; t < N / TILE; t++) {
//         As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE + threadIdx.x];
//         Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
//         __syncthreads();

//         for (int k = 0; k < TILE; k++)
//             sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
//         __syncthreads();
//     }

//     C[row * N + col] = sum;https://zhihaojia.medium.com/compiling-llms-into-a-megakernel-a-path-to-low-latency-inference-cf7840913c17
// }

// // ---------------------------------------------------------------
// // MAIN: run and print simple demos
// // ---------------------------------------------------------------

// int main() {
//     int N = 256;

//     // ---------------------------
//     // Host memory
//     // ---------------------------
//     float *h_a = new float[N];
//     float *h_b = new float[N];
//     float *h_out = new float[N];

//     for (int i = 0; i < N; i++) {
//         h_a[i] = float(i);
//         h_b[i] = float(2*i);
//     }

//     // ---------------------------
//     // Device memory
//     // ---------------------------
//     float *d_a, *d_b, *d_out;
//     CHECK(cudaMalloc(&d_a, N * sizeof(float)));
//     CHECK(cudaMalloc(&d_b, N * sizeof(float)));
//     CHECK(cudaMalloc(&d_out, N * sizeof(float)));

//     CHECK(cudaMemcpy(d_a, h_a, N*sizeof(float), cudaMemcpyHostToDevice));
//     CHECK(cudaMemcpy(d_b, h_b, N*sizeof(float), cudaMemcpyHostToDevice));

//     // ---------------------------
//     // Launch add kernel
//     // ---------------------------
//     dim3 block(256);
//     dim3 grid((N + block.x - 1) / block.x);
//     add_kernel<<<grid, block>>>(d_a, d_b, d_out, N);
//     CHECK(cudaDeviceSynchronize());

//     CHECK(cudaMemcpy(h_out, d_out, N*sizeof(float), cudaMemcpyDeviceToHost));

//     std::cout << "Add kernel output example: "
//               << h_out[0] << ", " << h_out[1] << ", " << h_out[2] << std::endl;

//     // ---------------------------
//     // Shared memory example
//     // ---------------------------
//     shared_memory_demo<<<1, 256, 256*sizeof(float)>>>(d_a, d_out);
//     CHECK(cudaDeviceSynchronize());
//     CHECK(cudaMemcpy(h_out, d_out, N*sizeof(float), cudaMemcpyDeviceToHost));

//     std::cout << "Shared memory example: " << h_out[10] << std::endl;

//     // ---------------------------
//     // Warp reduction example
//     // ---------------------------
//     warp_reduce_kernel<<<1, 256>>>(d_a, d_out);
//     CHECK(cudaDeviceSynchronize());
//     float h_warp[8];
//     CHECK(cudaMemcpy(h_warp, d_out, 8 * sizeof(float), cudaMemcpyDeviceToHost));

//     std::cout << "Warp reduction partial sums: ";
//     for (int i = 0; i < 8; i++) std::cout << h_warp[i] << " ";
//     std::cout << std::endl;

//     // ---------------------------
//     // Block reduction example
//     // ---------------------------
//     block_reduce_kernel<<<1, 256, 256*sizeof(float)>>>(d_a, d_out);
//     CHECK(cudaDeviceSynchronize());
//     CHECK(cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));
//     std::cout << "Block reduction total sum: " << h_out[0] << std::endl;

//     // ---------------------------
//     // Matrix multiply tiled (16x16)
//     // ---------------------------
//     const int M = 16 * 16;
//     float *h_A = new float[M*M];
//     float *h_B = new float[M*M];
//     float *h_C = new float[M*M];

//     for (int i = 0; i < M*M; i++) {
//         h_A[i] = 1.f;
//         h_B[i] = 2.f;
//     }

//     float *d_A, *d_B, *d_C;
//     CHECK(cudaMalloc(&d_A, M*M*sizeof(float)));
//     CHECK(cudaMalloc(&d_B, M*M*sizeof(float)));
//     CHECK(cudaMalloc(&d_C, M*M*sizeof(float)));

//     CHECK(cudaMemcpy(d_A, h_A, M*M*sizeof(float), cudaMemcpyHostToDevice));
//     CHECK(cudaMemcpy(d_B, h_B, M*M*sizeof(float), cudaMemcpyHostToDevice));

//     dim3 block2(TILE, TILE);
//     dim3 grid2(M / TILE, M / TILE);

//     matmul_tiled<<<grid2, block2>>>(d_A, d_B, d_C, M);
//     CHECK(cudaDeviceSynchronize());
//     CHECK(cudaMemcpy(h_C, d_C, M*M*sizeof(float), cudaMemcpyDeviceToHost));

//     std::cout << "Matmul example C[0]: " << h_C[0] << std::endl;

//     return 0;
// }
