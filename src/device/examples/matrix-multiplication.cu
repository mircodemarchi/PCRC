/**
 * @file matrix-multiplication.cu
 * @date 27/07/2020
 * @author Mirco De Marchi
 * @brief Source of matrix multiplication parallel and sequential algorithms.
 */

#include "matrix-multiplication.cuh"

#include <chrono>
#include <random>

#include "Timer.cuh"
#include "CheckError.cuh"
//------------------------------------------------------------------------------

#define BLOCK_SIZE 32   ///< Thread block dim.
//------------------------------------------------------------------------------

/**
 * @brief Device Matrix Multiplication kernel executed by each GPU thread.
 * @param d_matrixA Matrix A input param.
 * @param d_matrixB Matrix B input param.
 * @param d_matrixC Matric C = A * B output param.
 */
__global__
static void matrixMultiplicationKernel(const uint32_t* d_matrixA,
                                       const uint32_t* d_matrixB,
                                       uint32_t*       d_matrixC,
                                       const size_t N);
//------------------------------------------------------------------------------

void *matrix_mul_init_common(const constants_t *c)
{
    const size_t N = c->N;

    // Host allocation.
    matrix_mul_params_t *params = new matrix_mul_params_t;
    params->h_matrix_a          = new uint32_t[N * N];
    params->h_matrix_b          = new uint32_t[N * N];
    params->h_matrix_res_host   = new uint32_t[N * N];
    params->h_matrix_res_dev    = new uint32_t[N * N];

    // Host initialization.
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<uint32_t> distribution(1, 100);

    for (size_t i = 0; i < N * N; i++) {
        params->h_matrix_a[i] = distribution(generator);
        params->h_matrix_b[i] = distribution(generator);
    }

    return (void *) params;
}

void *matrix_mul_init_device(const constants_t *c, void *params)
{
    const size_t N = c->N;
    matrix_mul_params_t *matrix_mul_params = (matrix_mul_params_t *) params;

    // Device allocation.
    SAFE_CALL( cudaMalloc( &matrix_mul_params->d_matrix_a  , sizeof(uint32_t) * N * N ) )
    SAFE_CALL( cudaMalloc( &matrix_mul_params->d_matrix_b  , sizeof(uint32_t) * N * N ) )
    SAFE_CALL( cudaMalloc( &matrix_mul_params->d_matrix_res, sizeof(uint32_t) * N * N ) )

    return params;
}

void *matrix_mul_init(const constants_t *c)
{
    return matrix_mul_init_device(c, matrix_mul_init_common(c));
}

void matrix_mul_sequential(const constants_t *c, void *params)
{
    const size_t N = c->N;

    uint32_t *h_matrixA = ((matrix_mul_params_t *) params)->h_matrix_a;
    uint32_t *h_matrixB = ((matrix_mul_params_t *) params)->h_matrix_b;
    uint32_t *h_matrixC = ((matrix_mul_params_t *) params)->h_matrix_res_host;

    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            uint32_t sum = 0;
            for (size_t k = 0; k < N; k++)
                 sum += h_matrixA[i * N + k] * h_matrixB[k * N + j];
            h_matrixC[i * N + j] = sum;
        }
    }
}

void matrix_mul_parallel(const constants_t *c, void *params)
{
    const size_t N = c->N;
    const size_t BLOCK_SIZE_X = BLOCK_SIZE;
    const size_t BLOCK_SIZE_Y = BLOCK_SIZE;

    uint32_t *d_matrixA = ((matrix_mul_params_t *) params)->d_matrix_a;
    uint32_t *d_matrixB = ((matrix_mul_params_t *) params)->d_matrix_b; 
    uint32_t *d_matrixC = ((matrix_mul_params_t *) params)->d_matrix_res;

    uint32_t *h_matrixA = ((matrix_mul_params_t *) params)->h_matrix_a;
    uint32_t *h_matrixB = ((matrix_mul_params_t *) params)->h_matrix_b;
    uint32_t *h_matrix_tmp = ((matrix_mul_params_t *) params)->h_matrix_res_dev;

    // Device copy inputs.
    SAFE_CALL( cudaMemcpy( d_matrixA, h_matrixA, sizeof(int) * N * N, 
        cudaMemcpyHostToDevice ) )
    SAFE_CALL( cudaMemcpy( d_matrixB, h_matrixB, sizeof(int) * N * N, 
        cudaMemcpyHostToDevice ) )

    // Device dim.
    dim3 DimGrid(N / BLOCK_SIZE_X, N / BLOCK_SIZE_Y, 1);
    if (N % BLOCK_SIZE_X) DimGrid.x++;
    if (N % BLOCK_SIZE_Y) DimGrid.y++;
    dim3 DimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);

    // Device kernel call.
    matrixMultiplicationKernel<<< DimGrid, DimBlock >>>(d_matrixA, d_matrixB, 
        d_matrixC, N);
    CHECK_CUDA_ERROR

    // Device copy result.
    SAFE_CALL( cudaMemcpy( h_matrix_tmp, d_matrixC, sizeof(int) * N * N, 
        cudaMemcpyDeviceToHost ) )
}

bool matrix_mul_compare(const constants_t *c, void *params)
{
    const size_t N = c->N;

    uint32_t *h_matrix_tmp   = ((matrix_mul_params_t *) params)->h_matrix_res_dev;
    uint32_t *h_matrixC      = ((matrix_mul_params_t *) params)->h_matrix_res_host;

    for (size_t i = 0; i < N * N; i++) {
        if (h_matrixC[i] != h_matrix_tmp[i]) {
            std::cerr << "wrong result at: ("
                      << (i / N) << ", " << (i % N) << ")"
                      << "\nhost:   " << h_matrixC[i]
                      << "\ndevice: " << h_matrix_tmp[i] << "\n\n";
            cudaDeviceReset();
            return false;
        }
    }
    return true;
}

void matrix_mul_free_common(void *params)
{
    uint32_t *h_matrixA      = ((matrix_mul_params_t *) params)->h_matrix_a;
    uint32_t *h_matrixB      = ((matrix_mul_params_t *) params)->h_matrix_b;
    uint32_t *h_matrixC      = ((matrix_mul_params_t *) params)->h_matrix_res_host;
    uint32_t *h_matrix_tmp   = ((matrix_mul_params_t *) params)->h_matrix_res_dev;

    // Free host.
    delete[] h_matrixA;
    delete[] h_matrixB;
    delete[] h_matrixC;
    delete[] h_matrix_tmp;
}

void matrix_mul_free_device(void *params)
{
    uint32_t *d_matrixA = ((matrix_mul_params_t *) params)->d_matrix_a;
    uint32_t *d_matrixB = ((matrix_mul_params_t *) params)->d_matrix_b; 
    uint32_t *d_matrixC = ((matrix_mul_params_t *) params)->d_matrix_res;

    // Free device.
    SAFE_CALL( cudaFree( d_matrixA ) )
    SAFE_CALL( cudaFree( d_matrixB ) )
    SAFE_CALL( cudaFree( d_matrixC ) )
}

void matrix_mul_free(void *params)
{
    // Free host.
    matrix_mul_free_common(params);
    // Free device.
    matrix_mul_free_device(params);
    // Free params.
    delete ((matrix_mul_params_t *) params);
}
//------------------------------------------------------------------------------

__global__
static void matrixMultiplicationKernel(const uint32_t* d_matrixA,
                                       const uint32_t* d_matrixB,
                                       uint32_t*       d_matrixC,
                                       const size_t N) {
    int i = threadIdx.y + (blockDim.y * blockIdx.y);
    int j = threadIdx.x + (blockDim.x * blockIdx.x);

    if (i < N && j < N) {
        size_t sum = 0;

        for (size_t k = 0; k < N; k++) {
            sum += d_matrixA[i * N + k] * d_matrixB[k * N + j];
        }
        
        d_matrixC[i * N + j] = sum;
    }
}