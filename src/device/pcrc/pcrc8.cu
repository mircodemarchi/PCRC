/**
 * @file pcrc8.cu
 * @date 01/08/2020
 * @author Mirco De Marchi
 * @brief Source of 8 bit CRC parallel and sequential algorithms.
 */

#include "pcrc8.cuh"

#include <chrono>
#include <random>
#include <arpa/inet.h>

#include "Timer.cuh"
#include "CheckError.cuh"

#include "crc8-bitwise.h"
#include "crc8-bytewise.h"
#include "mod2.h"
#include "mul2.h"

using namespace timer;
//------------------------------------------------------------------------------

#define M 1             ///< Size of CRC result.
//------------------------------------------------------------------------------

/**
 * @brief Device CRC8 kernel executed by each GPU thread.
 * @param d_message     Message from which calculate the CRC value.
 * @param d_beta        Array of beta factor.
 * @param d_generator   Polynomial generator.
 * @param d_crc         Pointer to the result of the CRC value.
 */
__global__
static void pcrc8_kernel(const uint8_t* d_message, 
                         const uint8_t* d_beta,
                         const uint16_t d_generator,
                         uint8_t *d_partial_crc);

__global__
static void pcrc8_kernel_reduction(const uint8_t* d_message, 
                                   const uint8_t* d_beta,
                                   const uint16_t d_generator,
                                   uint8_t *d_partial_crc);                          

//------------------------------------------------------------------------------

void *pcrc8_init_common(const constants_t *c)
{
    const size_t N = c->N;

    // Host allocation.
    pcrc8_params_t *params = new pcrc8_params_t;
    params->h_message      = new uint8_t[N];
    params->h_beta         = new uint8_t[N];
    params->h_crc_partial_res_dev = new uint8_t[CEIL(N, BLOCK_SIZE)];

    params->generator = CRC8_SAE;

    // Host initialization.
    params->h_crc_res_dev  = 0x00;
    params->h_crc_res_host = 0x00;

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<uint8_t> distribution(0x00, 0xFF);

    for (size_t i = 0; i < N; i++) 
    {
        params->h_message[i] = distribution(generator);
    }

    // Generate beta array.

    // Slow method.
    // for (size_t i = 0; i < N; i++) 
    // {
    //     size_t shift_buffer_length = M * (i + 1);
    //     uint8_t *shift_buffer = new uint8_t[shift_buffer_length + 1]();
    //     shift_buffer[0] = 0x01;
    //     params->h_beta[N - i - 1] = (uint8_t) mod2_16(shift_buffer, 
    //         shift_buffer_length + 1, params->generator + 0x100);
    //     delete[] shift_buffer;
    // }

    // Fast method.
    uint8_t *beta0 = new uint8_t[M + 1]();
    uint8_t mul16_arr[2] = {};
    beta0[0] = 0x01;
    params->h_beta[N/M - 1] = (uint32_t) mod2_16(beta0, M + 1, 
                              params->generator + 0x100);
    delete[] beta0;
    for (size_t i = 1; i < N/M; i++) 
    {
        uint16_t mul16 = mul2_16(params->h_beta[N/M - i], params->h_beta[N/M - 1]);
        mul16 = htobe16(mul16);
        memcpy(mul16_arr, &mul16, 2);
        params->h_beta[N/M - i - 1] = (uint8_t) mod2_16(mul16_arr, 2, 
            params->generator + 0x100);
    }

    return (void *) params;
}

void *pcrc8_init_device(const constants_t *c, void *params)
{
    const size_t N = c->N;
    pcrc8_params_t * pcrc8_params = ((pcrc8_params_t *) params);

    // Reset tmp data.
    pcrc8_params->h_crc_res_dev  = 0x00;
    pcrc8_params->h_crc_res_host = 0x00;

    // Device allocation.
    SAFE_CALL(cudaMalloc(&pcrc8_params->d_message , sizeof(uint8_t) * N))
    SAFE_CALL(cudaMalloc(&pcrc8_params->d_beta    , sizeof(uint8_t) * N))
    SAFE_CALL(cudaMalloc(&pcrc8_params->d_crc_partial_res, 
                         sizeof(uint8_t) * CEIL(N, BLOCK_SIZE)))

    return params;
}

void *pcrc8_init_device_reduction(const constants_t *c, void *params)
{
    return pcrc8_init_device(c, params);
}

void *pcrc8_init_device_task_parallelism(const constants_t *c, void *params)
{
    const uint8_t STREAM_DIM = c->STREAM_DIM;
    const size_t  SEG_SIZE   = c->SEG_SIZE;
    pcrc8_params_t * pcrc8_params = ((pcrc8_params_t *) params);

    // Reset tmp data.
    pcrc8_params->h_crc_res_dev  = 0x00;
    pcrc8_params->h_crc_res_host = 0x00;

    // Device allocation.
    SAFE_CALL(cudaMalloc(&pcrc8_params->d_message, 
              sizeof(uint8_t) * SEG_SIZE * STREAM_DIM))
    SAFE_CALL(cudaMalloc(&pcrc8_params->d_beta, 
              sizeof(uint8_t) * SEG_SIZE * STREAM_DIM))
    SAFE_CALL(cudaMalloc(&pcrc8_params->d_crc_partial_res, 
              sizeof(uint8_t) * CEIL(SEG_SIZE, BLOCK_SIZE) * STREAM_DIM))

    return params;
}

void *pcrc8_init(const constants_t *c)
{
    return pcrc8_init_device(c, pcrc8_init_common(c));
}

void *pcrc8_init_reduction(const constants_t *c)
{
    return pcrc8_init_device_reduction(c, pcrc8_init_common(c));
}

void *pcrc8_init_task_parallelism(const constants_t *c)
{
    return pcrc8_init_device_task_parallelism(c, pcrc8_init_common(c));
}

void pcrc8_sequential(const constants_t *c, void *params, host_time_t *h_time)
{
    Timer<HOST> TM_host;

    const size_t N = c->N;

    uint8_t *message  = ((pcrc8_params_t *) params)->h_message;
    uint8_t generator = ((pcrc8_params_t *) params)->generator;
    // TODO: implement crc8 with generator.

    TM_host.start();
    uint8_t crc = crc8_bitwise(message, N);
    ((pcrc8_params_t *) params)->h_crc_res_host = crc;
    TM_host.stop();

    h_time->is_initialized = true;
    h_time->exec_time = TM_host.duration();
}

void pcrc8_sequential_bytewise(const constants_t *c, void *params, host_time_t *h_time)
{
    Timer<HOST> TM_host;

    const size_t N = c->N;

    uint8_t *message  = ((pcrc8_params_t *) params)->h_message;
    uint8_t generator = ((pcrc8_params_t *) params)->generator;
    // TODO: implement crc8 with generator.

    TM_host.start();
    uint8_t crc = crc8_bytewise(message, N, crc8_lu);
    ((pcrc8_params_t *) params)->h_crc_res_host = crc;
    TM_host.stop();

    h_time->is_initialized = true;
    h_time->exec_time = TM_host.duration();
}

void pcrc8_parallel(const constants_t *c, void *params, device_time_t *d_time)
{
    Timer<DEVICE> TM_device_kernel, TM_device_htod, TM_device_dtoh;
    Timer<HOST> TM_host;

    const size_t N = c->N;

    uint8_t *d_message = ((pcrc8_params_t *) params)->d_message;
    uint8_t *d_beta    = ((pcrc8_params_t *) params)->d_beta; 
    uint8_t *d_crc_partial_res = ((pcrc8_params_t *) params)->d_crc_partial_res;

    uint8_t *h_message = ((pcrc8_params_t *) params)->h_message;
    uint8_t *h_beta    = ((pcrc8_params_t *) params)->h_beta;
    uint8_t generator = ((pcrc8_params_t *) params)->generator;
    uint8_t *h_crc_partial_res_dev = 
        ((pcrc8_params_t *) params)->h_crc_partial_res_dev;

    // Device copy inputs.
    TM_device_htod.start();
    SAFE_CALL(cudaMemcpy(d_message, h_message, sizeof(uint8_t) * N, 
        cudaMemcpyHostToDevice))
    SAFE_CALL(cudaMemcpy(d_beta, h_beta, sizeof(uint8_t) * N, 
        cudaMemcpyHostToDevice))
    TM_device_htod.stop();

    TM_device_kernel.start();
    // Device dim.
    dim3 DimGrid(N / BLOCK_SIZE, 1, 1);
    if (N % BLOCK_SIZE) DimGrid.x++;
    dim3 DimBlock(BLOCK_SIZE, 1, 1);

    // Device kernel call.
    pcrc8_kernel<<< DimGrid, DimBlock >>>(d_message, d_beta, 
        (uint16_t) generator + 0x100, d_crc_partial_res);
    CHECK_CUDA_ERROR
    TM_device_kernel.stop();

    // Device copy result.
    TM_device_dtoh.start();
    SAFE_CALL(cudaMemcpy(h_crc_partial_res_dev, d_crc_partial_res, 
        sizeof(uint8_t) * CEIL(N, BLOCK_SIZE), cudaMemcpyDeviceToHost))
    TM_device_dtoh.stop();
    
    TM_host.start();
    for (size_t i = 0; i < CEIL(N, BLOCK_SIZE); i++) 
    {
        ((pcrc8_params_t *) params)->h_crc_res_dev ^= h_crc_partial_res_dev[i];
    }
    TM_host.stop();

    d_time->is_initialized      = true;
    d_time->is_task_parallelism = false;
    d_time->htod_time   = TM_device_htod.duration();
    d_time->kernel_time = TM_device_kernel.duration() + TM_host.duration();
    d_time->dtoh_time   = TM_device_dtoh.duration();
}

void pcrc8_parallel_reduction(const constants_t *c, void *params, device_time_t *d_time)
{
    Timer<DEVICE> TM_device_kernel, TM_device_htod, TM_device_dtoh;
    Timer<HOST> TM_host;

    const size_t N = c->N;

    uint8_t *d_message = ((pcrc8_params_t *) params)->d_message;
    uint8_t *d_beta    = ((pcrc8_params_t *) params)->d_beta; 
    uint8_t *d_crc_partial_res = ((pcrc8_params_t *) params)->d_crc_partial_res;

    uint8_t *h_message = ((pcrc8_params_t *) params)->h_message;
    uint8_t *h_beta    = ((pcrc8_params_t *) params)->h_beta;
    uint8_t generator = ((pcrc8_params_t *) params)->generator;
    uint8_t *h_crc_partial_res_dev = 
        ((pcrc8_params_t *) params)->h_crc_partial_res_dev;

    // Device copy inputs.
    TM_device_htod.start();
    SAFE_CALL(cudaMemcpy(d_message, h_message, sizeof(uint8_t) * N, 
        cudaMemcpyHostToDevice))
    SAFE_CALL(cudaMemcpy(d_beta, h_beta, sizeof(uint8_t) * N, 
        cudaMemcpyHostToDevice))
    TM_device_htod.stop();

    TM_device_kernel.start();
    // Device dim.
    dim3 DimGrid(N / BLOCK_SIZE, 1, 1);
    if (N % BLOCK_SIZE) DimGrid.x++;
    dim3 DimBlock(BLOCK_SIZE, 1, 1);

    // Device kernel call.
    pcrc8_kernel_reduction<<< DimGrid, DimBlock >>>(d_message, d_beta, 
        (uint16_t) generator + 0x100, d_crc_partial_res);
    CHECK_CUDA_ERROR
    TM_device_kernel.stop();

    // Device copy result.
    TM_device_dtoh.start();
    SAFE_CALL(cudaMemcpy(h_crc_partial_res_dev, d_crc_partial_res, 
        sizeof(uint8_t) * CEIL(N, BLOCK_SIZE), cudaMemcpyDeviceToHost))
    TM_device_dtoh.stop();
    
    TM_host.start();
    for (size_t i = 0; i < CEIL(N, BLOCK_SIZE); i++) 
    {
        ((pcrc8_params_t *) params)->h_crc_res_dev ^= h_crc_partial_res_dev[i];
    }
    TM_host.stop();

    d_time->is_initialized      = true;
    d_time->is_task_parallelism = false;
    d_time->htod_time   = TM_device_htod.duration();
    d_time->kernel_time = TM_device_kernel.duration() + TM_host.duration();
    d_time->dtoh_time   = TM_device_dtoh.duration();
}

void pcrc8_parallel_task_parallelism(const constants_t *c, void *params, device_time_t *d_time)
{
    Timer<DEVICE> TM_device;
    Timer<HOST> TM_host;

    const size_t  N          = c->N;
    const uint8_t STREAM_DIM = c->STREAM_DIM;
    const size_t  SEG_SIZE   = c->SEG_SIZE;

    uint8_t *d_message = ((pcrc8_params_t *) params)->d_message;
    uint8_t *d_beta    = ((pcrc8_params_t *) params)->d_beta; 
    uint8_t *d_crc_partial_res = ((pcrc8_params_t *) params)->d_crc_partial_res;

    uint8_t *h_message = ((pcrc8_params_t *) params)->h_message;
    uint8_t *h_beta    = ((pcrc8_params_t *) params)->h_beta;
    uint8_t generator = ((pcrc8_params_t *) params)->generator;
    uint8_t *h_crc_partial_res_dev = 
        ((pcrc8_params_t *) params)->h_crc_partial_res_dev;

    // TASK PARALLELISM
    TM_device.start();
    cudaStream_t stream[STREAM_DIM];
    for (uint8_t i = 0; i < STREAM_DIM; i++) {
        cudaStreamCreate(stream + i);
    }

    // Reminder: each STREAM takes one SEGMENT.
    for (int i = 0; i < N; i += SEG_SIZE * STREAM_DIM) {
        // 1. Copy inputs for every streams.
        for (int stream_index = 0; stream_index < STREAM_DIM; stream_index++) {
            int h_input_offset = i + (stream_index * SEG_SIZE);
            int d_input_offset = stream_index * SEG_SIZE;
            SAFE_CALL( 
                cudaMemcpyAsync(
                    d_message + d_input_offset, 
                    h_message + h_input_offset, 
                    sizeof(uint8_t) * SEG_SIZE, 
                    cudaMemcpyHostToDevice, 
                    stream[stream_index]) 
            )
            SAFE_CALL( 
                cudaMemcpyAsync(
                    d_beta + d_input_offset, 
                    h_beta + h_input_offset, 
                    sizeof(uint8_t) * SEG_SIZE, 
                    cudaMemcpyHostToDevice, 
                    stream[stream_index]) 
            )
        }

        // 2. Call kernels for every streams.
        for (int stream_index = 0; stream_index < STREAM_DIM; stream_index++) {
            int d_output_offset = stream_index * CEIL(SEG_SIZE, BLOCK_SIZE);
            int d_input_offset = stream_index * SEG_SIZE;
            pcrc8_kernel_reduction<<< SEG_SIZE / BLOCK_SIZE, BLOCK_SIZE, 0, stream[stream_index]>>>
                (d_message + d_input_offset, d_beta + d_input_offset, (uint16_t) generator + 0x100, d_crc_partial_res + d_output_offset);
        }

        // 3. Copy outputs for every streams.
        for (int stream_index = 0; stream_index < STREAM_DIM; stream_index++) {
            int h_output_offset = ((i / SEG_SIZE) + stream_index) * CEIL(SEG_SIZE, BLOCK_SIZE);
            int d_output_offset = stream_index * CEIL(SEG_SIZE, BLOCK_SIZE);
            SAFE_CALL( 
                cudaMemcpyAsync( 
                    h_crc_partial_res_dev + h_output_offset, 
                    d_crc_partial_res + d_output_offset, 
                    sizeof(uint8_t) * CEIL(SEG_SIZE, BLOCK_SIZE), 
                    cudaMemcpyDeviceToHost,
                    stream[stream_index]) )
        }
    }
    TM_device.stop();

    TM_host.start();
    for (size_t i = 0; i < CEIL(N, BLOCK_SIZE); i++) 
    {
        ((pcrc8_params_t *) params)->h_crc_res_dev ^= h_crc_partial_res_dev[i];
    }
    TM_host.stop();

    d_time->is_initialized      = true;
    d_time->is_task_parallelism = true;
    d_time->kernel_time = TM_device.duration() + TM_host.duration();
}

bool pcrc8_compare(const constants_t *c, void *params)
{
    uint8_t h_crc_res_dev  = ((pcrc8_params_t *) params)->h_crc_res_dev;
    uint8_t h_crc_res_host = ((pcrc8_params_t *) params)->h_crc_res_host;

    return h_crc_res_dev == h_crc_res_host;
}

bool pcrc8_compare_reduction(const constants_t *c, void *params)
{
    return pcrc8_compare(c, params);
}

bool pcrc8_compare_task_parallelism(const constants_t *c, void *params)
{
    return pcrc8_compare(c, params);
}

void pcrc8_free_common(void *params)
{
    uint8_t *h_message = ((pcrc8_params_t *) params)->h_message;
    uint8_t *h_beta    = ((pcrc8_params_t *) params)->h_beta;
    uint8_t *h_crc_partial_res_dev = 
        ((pcrc8_params_t *) params)->h_crc_partial_res_dev;

    // Free host.
    delete[] h_message;
    delete[] h_beta;
    delete[] h_crc_partial_res_dev;
}

void pcrc8_free_device(void *params)
{
    uint8_t *d_message = ((pcrc8_params_t *) params)->d_message;
    uint8_t *d_beta    = ((pcrc8_params_t *) params)->d_beta; 
    uint8_t *d_crc_partial_res = ((pcrc8_params_t *) params)->d_crc_partial_res;

    SAFE_CALL(cudaFree(d_message))
    SAFE_CALL(cudaFree(d_beta))
    SAFE_CALL(cudaFree(d_crc_partial_res))
}

void pcrc8_free_device_reduction(void *params)
{
    pcrc8_free_device(params);
}

void pcrc8_free_device_task_parallelism(void *params)
{
    pcrc8_free_device(params);
}

void pcrc8_free(void *params)
{
    // Free host.
    pcrc8_free_common(params);
    // Free device.
    pcrc8_free_device(params);
    // Free params.
    delete ((pcrc8_params_t *) params);
}

void pcrc8_free_reduction(void *params) 
{
    pcrc8_free(params);
}

void pcrc8_free_task_parallelism(void *params) 
{
    pcrc8_free(params);
}
//------------------------------------------------------------------------------

__global__
static void pcrc8_kernel(const uint8_t* d_message, 
                         const uint8_t* d_beta,
                         const uint16_t d_generator,
                         uint8_t *d_partial_crc)
{
    // __shared__ uint8_t ds_mem_message[BLOCK_SIZE];
    // __shared__ uint8_t ds_mem_beta[BLOCK_SIZE];
    __shared__ uint8_t ds_mem_crc[BLOCK_SIZE];
    uint32_t globalIndex = threadIdx.x + blockDim.x * blockIdx.x;

    // ds_mem_message[threadIdx.x] = d_message[globalIndex];
    // ds_mem_beta[threadIdx.x] = d_beta[globalIndex];
    uint8_t w    = d_message[globalIndex];
    uint8_t beta = d_beta[globalIndex];
    // __syncthreads();

    // Binary modulo 2 multiplication between w and beta.
    uint16_t mul = 0;
    for(uint8_t i = 0; i < (sizeof(uint8_t) * 8); i++)
    {
        if (beta & (1U << i))
        {
            mul ^= (uint16_t) w << i;
        }
    }

    uint16_t ret = 0;

    // Compute division of mul result by polynomial generator value.
    for (uint8_t i = 0; i < sizeof(uint16_t) * 8; i++)
    {
        // Compute subtraction.
        if ((ret & 0x0100) != 0)
        {
            ret = (uint16_t)(ret ^ d_generator);
        }

        // Shift by 1 all the divident buffer.
        ret = (uint16_t)(ret << 1) 
            | (0x0001 & (mul >> (sizeof(uint16_t) * 8 - i - 1)));
    }

    // Compute the last subtraction.
    if ((ret & 0x0100) != 0)
    {
        ret = (uint16_t)(ret ^ d_generator);
    }

    ds_mem_crc[threadIdx.x] = (uint8_t) ret;
    __syncthreads();

    if (threadIdx.x == 0) 
    {
        uint8_t partial_crc = 0;
        for (size_t i = 0; i < BLOCK_SIZE; i++)
        {
            partial_crc ^= ds_mem_crc[i];
        }
        d_partial_crc[blockIdx.x] = partial_crc;
    }
}

__global__
static void pcrc8_kernel_reduction(const uint8_t* d_message, 
                                   const uint8_t* d_beta,
                                   const uint16_t d_generator,
                                   uint8_t *d_partial_crc)
{
    // __shared__ uint8_t ds_mem_message[BLOCK_SIZE];
    // __shared__ uint8_t ds_mem_beta[BLOCK_SIZE];
    __shared__ uint8_t ds_mem_crc[BLOCK_SIZE];
    uint32_t globalIndex = threadIdx.x + blockDim.x * blockIdx.x;

    // ds_mem_message[threadIdx.x] = d_message[globalIndex];
    // ds_mem_beta[threadIdx.x] = d_beta[globalIndex];
    uint8_t w    = d_message[globalIndex];
    uint8_t beta = d_beta[globalIndex];
    // __syncthreads();

    // Binary modulo 2 multiplication between w and beta.
    uint16_t mul = 0;
    for(uint8_t i = 0; i < (sizeof(uint8_t) * 8); i++)
    {
        if (beta & (1U << i))
        {
            mul ^= (uint16_t) w << i;
        }
    }

    uint16_t ret = 0;

    // Compute division of mul result by polynomial generator value.
    for (uint8_t i = 0; i < sizeof(uint16_t) * 8; i++)
    {
        // Compute subtraction.
        if ((ret & 0x0100) != 0)
        {
            ret = (uint16_t)(ret ^ d_generator);
        }

        // Shift by 1 all the divident buffer.
        ret = (uint16_t)(ret << 1) 
            | (0x0001 & (mul >> (sizeof(uint16_t) * 8 - i - 1)));
    }

    // Compute the last subtraction.
    if ((ret & 0x0100) != 0)
    {
        ret = (uint16_t)(ret ^ d_generator);
    }

    ds_mem_crc[threadIdx.x] = (uint8_t) ret;
    __syncthreads();

    // Perform Reduction.
    for (size_t i = 1; i < blockDim.x; i *= 2) {
        size_t index = threadIdx.x * i * 2;
        if (index < blockDim.x) {
            ds_mem_crc[index] ^= ds_mem_crc[index + i]; 
        }

        __syncthreads();
    }

    // Write back in memory.
    if (threadIdx.x == 0) {
        d_partial_crc[blockIdx.x] = ds_mem_crc[0];
    }
}
