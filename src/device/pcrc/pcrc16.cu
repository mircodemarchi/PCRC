/**
 * @file pcrc16.cu
 * @date 08/09/2020
 * @author Mirco De Marchi
 * @brief Source of 16 bit CRC parallel and sequential algorithms.
 */

#include "pcrc16.cuh"

#include <chrono>
#include <random>
#include <arpa/inet.h>

#include "Timer.cuh"
#include "CheckError.cuh"

#include "crc16-bitwise.h"
#include "crc16-bytewise.h"
#include "mod2.h"
#include "mul2.h"

using namespace timer;
//------------------------------------------------------------------------------

#define M 2             ///< Size of CRC result.
//------------------------------------------------------------------------------

/**
 * @brief Device CRC16 kernel executed by each GPU thread.
 * @param d_message     Message from which calculate the CRC value.
 * @param d_beta        Array of beta factor.
 * @param d_generator   Polynomial generator.
 * @param d_crc         Pointer to the result of the CRC value.
 */
__global__
static void pcrc16_kernel(const uint16_t* d_message, 
                          const uint16_t* d_beta,
                          const uint32_t d_generator,
                          uint16_t *d_partial_crc);

__global__
static void pcrc16_kernel_reduction(const uint16_t* d_message, 
                                    const uint16_t* d_beta,
                                    const uint32_t d_generator,
                                    uint16_t *d_partial_crc);                          

//------------------------------------------------------------------------------

void *pcrc16_init_common(const constants_t *c)
{
    const size_t N = c->N;

    // Host allocation.
    pcrc16_params_t *params = new pcrc16_params_t;
    params->h_message      = new uint8_t[N];
    params->h_message_16   = new uint16_t[N/M];
    params->h_beta         = new uint16_t[N/M];
    params->h_crc_partial_res_dev = new uint16_t[CEIL(N, BLOCK_SIZE)/M];

    params->generator = CRC16_CCITT;

    // Host initialization.
    params->h_crc_res_dev  = 0x0000;
    params->h_crc_res_host = 0x0000;

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<uint8_t> distribution(0x00, 0xFF);

    for (size_t i = 0; i < N; i++) 
    {
        params->h_message[i] = distribution(generator);
    }

    // Convert h_message in h_message_16.
    memcpy(params->h_message_16, params->h_message, N);
    for (size_t i = 0; i < N/M; i++)
    {
        params->h_message_16[i] = ntohs(params->h_message_16[i]);
    }

    // Generate beta array.

    // Slow method.
    // for (size_t i = 0; i < N/M; i++) 
    // {
    //     size_t shift_buffer_length = M * (i + 1);
    //     uint8_t *shift_buffer = new uint8_t[shift_buffer_length + 1]();
    //     shift_buffer[0] = 0x01;
    //     params->h_beta[N/M - i - 1] = (uint16_t) mod2_32(
    //         shift_buffer, shift_buffer_length + 1, params->generator + 0x10000);
    //     delete[] shift_buffer;
    // }

    // Fast method.
    uint8_t *beta0 = new uint8_t[M + 1]();
    uint8_t mul32_arr[4] = {};
    beta0[0] = 0x01;
    params->h_beta[N/M - 1] = (uint16_t) mod2_32(beta0, M + 1, 
                              params->generator + 0x10000);
    delete[] beta0;
    for (size_t i = 1; i < N/M; i++) 
    {
        uint32_t mul32 = mul2_16(params->h_beta[N/M - i], params->h_beta[N/M - 1]);
        mul32 = htobe32(mul32);
        memcpy(mul32_arr, &mul32, 4);
        params->h_beta[N/M - i - 1] = (uint16_t) mod2_32(
            mul32_arr, 4, params->generator + 0x10000);
    }

    return (void *) params;
}

void *pcrc16_init_device(const constants_t *c, void *params)
{
    const size_t N = c->N;
    pcrc16_params_t *pcrc16_params = (pcrc16_params_t *) params;

    // Reset tmp data.
    pcrc16_params->h_crc_res_dev  = 0x00;
    pcrc16_params->h_crc_res_host = 0x00;

    // Device allocation.
    SAFE_CALL(cudaMalloc(&pcrc16_params->d_message , sizeof(uint16_t) * N / M))
    SAFE_CALL(cudaMalloc(&pcrc16_params->d_beta    , sizeof(uint16_t) * N / M))
    SAFE_CALL(cudaMalloc(&pcrc16_params->d_crc_partial_res, 
                         sizeof(uint16_t) * CEIL(N, BLOCK_SIZE) / M))

    return params;
}

void *pcrc16_init_device_reduction(const constants_t *c, void *params)
{
    return pcrc16_init_device(c, params);
}

void *pcrc16_init_device_task_parallelism(const constants_t *c, void *params)
{
    const uint8_t STREAM_DIM = c->STREAM_DIM;
    const size_t  SEG_SIZE   = c->SEG_SIZE;
    pcrc16_params_t *pcrc16_params = (pcrc16_params_t *) params;

    // Reset tmp data.
    pcrc16_params->h_crc_res_dev  = 0x00;
    pcrc16_params->h_crc_res_host = 0x00;
    
    // Device allocation.
    SAFE_CALL(cudaMalloc(&pcrc16_params->d_message, 
              sizeof(uint16_t) * SEG_SIZE * STREAM_DIM / M))
    SAFE_CALL(cudaMalloc(&pcrc16_params->d_beta, 
              sizeof(uint16_t) * SEG_SIZE * STREAM_DIM / M))
    SAFE_CALL(cudaMalloc(&pcrc16_params->d_crc_partial_res, 
              sizeof(uint16_t) * CEIL(SEG_SIZE, BLOCK_SIZE) * STREAM_DIM / M))

    return params;
}

void *pcrc16_init(const constants_t *c)
{
    return pcrc16_init_device(c, pcrc16_init_common(c));
}

void *pcrc16_init_reduction(const constants_t *c)
{
    return pcrc16_init_device_reduction(c, pcrc16_init_common(c));
}

void *pcrc16_init_task_parallelism(const constants_t *c)
{
    return pcrc16_init_device_task_parallelism(c, pcrc16_init_common(c));
}

void pcrc16_sequential(const constants_t *c, void *params, host_time_t *h_time)
{
    Timer<HOST> TM_host;

    const size_t N = c->N;

    uint8_t *message  = ((pcrc16_params_t *) params)->h_message;
    uint16_t generator = ((pcrc16_params_t *) params)->generator;
    // TODO: implement crc16 with generator.

    TM_host.start();
    uint16_t crc = crc16_bitwise(message, N);
    ((pcrc16_params_t *) params)->h_crc_res_host = crc;
    TM_host.stop();

    h_time->is_initialized = true;
    h_time->exec_time = TM_host.duration();
}

void pcrc16_sequential_bytewise(const constants_t *c, void *params, host_time_t *h_time)
{
    Timer<HOST> TM_host;

    const size_t N = c->N;

    uint8_t *message  = ((pcrc16_params_t *) params)->h_message;
    uint16_t generator = ((pcrc16_params_t *) params)->generator;
    // TODO: implement crc16 with generator.

    TM_host.start();
    uint16_t crc = crc16_bytewise(message, N, crc16_lu);
    ((pcrc16_params_t *) params)->h_crc_res_host = crc;
    TM_host.stop();

    h_time->is_initialized = true;
    h_time->exec_time = TM_host.duration();
}

void pcrc16_parallel(const constants_t *c, void *params, device_time_t *d_time)
{
    Timer<DEVICE> TM_device_kernel, TM_device_htod, TM_device_dtoh;
    Timer<HOST> TM_host;

    const size_t N = c->N;

    uint16_t *d_message = ((pcrc16_params_t *) params)->d_message;
    uint16_t *d_beta    = ((pcrc16_params_t *) params)->d_beta; 
    uint16_t *d_crc_partial_res = ((pcrc16_params_t *) params)->d_crc_partial_res;

    uint16_t *h_message = ((pcrc16_params_t *) params)->h_message_16;
    uint16_t *h_beta    = ((pcrc16_params_t *) params)->h_beta;
    uint16_t generator = ((pcrc16_params_t *) params)->generator;
    uint16_t *h_crc_partial_res_dev = 
        ((pcrc16_params_t *) params)->h_crc_partial_res_dev;

    // Device copy inputs.
    TM_device_htod.start();
    SAFE_CALL(cudaMemcpy(d_message, h_message, sizeof(uint16_t) * N/M, 
        cudaMemcpyHostToDevice))
    SAFE_CALL(cudaMemcpy(d_beta, h_beta, sizeof(uint16_t) * N/M, 
        cudaMemcpyHostToDevice))
    TM_device_htod.stop();

    TM_device_kernel.start();
    // Device dim.
    dim3 DimGrid((N/M) / BLOCK_SIZE, 1, 1);
    if ((N/M) % BLOCK_SIZE) DimGrid.x++;
    dim3 DimBlock(BLOCK_SIZE, 1, 1);

    // Device kernel call.
    pcrc16_kernel<<< DimGrid, DimBlock >>>(d_message, d_beta, 
        (uint32_t) generator + 0x10000, d_crc_partial_res);
    CHECK_CUDA_ERROR
    TM_device_kernel.stop();

    // Device copy result.
    TM_device_dtoh.start();
    SAFE_CALL(cudaMemcpy(h_crc_partial_res_dev, d_crc_partial_res, 
        sizeof(uint16_t) * CEIL(N, BLOCK_SIZE) / M, cudaMemcpyDeviceToHost))
    TM_device_dtoh.stop();
    
    TM_host.start();
    for (size_t i = 0; i < (CEIL(N, BLOCK_SIZE) / M); i++)
    {
        ((pcrc16_params_t *) params)->h_crc_res_dev ^= h_crc_partial_res_dev[i];
    }
    TM_host.stop();

    d_time->is_initialized      = true;
    d_time->is_task_parallelism = false;
    d_time->htod_time   = TM_device_htod.duration();
    d_time->kernel_time = TM_device_kernel.duration() + TM_host.duration();
    d_time->dtoh_time   = TM_device_dtoh.duration();
}

void pcrc16_parallel_reduction(const constants_t *c, void *params, device_time_t *d_time)
{
    Timer<DEVICE> TM_device_kernel, TM_device_htod, TM_device_dtoh;
    Timer<HOST> TM_host;

    const size_t N = c->N;

    uint16_t *d_message = ((pcrc16_params_t *) params)->d_message;
    uint16_t *d_beta    = ((pcrc16_params_t *) params)->d_beta; 
    uint16_t *d_crc_partial_res = ((pcrc16_params_t *) params)->d_crc_partial_res;

    uint16_t *h_message = ((pcrc16_params_t *) params)->h_message_16;
    uint16_t *h_beta    = ((pcrc16_params_t *) params)->h_beta;
    uint16_t generator = ((pcrc16_params_t *) params)->generator;
    uint16_t *h_crc_partial_res_dev = 
        ((pcrc16_params_t *) params)->h_crc_partial_res_dev;

    // Device copy inputs.
    TM_device_htod.start();
    SAFE_CALL(cudaMemcpy(d_message, h_message, sizeof(uint16_t) * N/M, 
        cudaMemcpyHostToDevice))
    SAFE_CALL(cudaMemcpy(d_beta, h_beta, sizeof(uint16_t) * N/M, 
        cudaMemcpyHostToDevice))
    TM_device_htod.stop();

    TM_device_kernel.start();
    // Device dim.
    dim3 DimGrid((N/M) / BLOCK_SIZE, 1, 1);
    if ((N/M) % BLOCK_SIZE) DimGrid.x++;
    dim3 DimBlock(BLOCK_SIZE, 1, 1);

    // Device kernel call.
    pcrc16_kernel_reduction<<< DimGrid, DimBlock >>>(d_message, d_beta, 
        (uint32_t) generator + 0x10000, d_crc_partial_res);
    CHECK_CUDA_ERROR
    TM_device_kernel.stop();

    // Device copy result.
    TM_device_dtoh.start();
    SAFE_CALL(cudaMemcpy(h_crc_partial_res_dev, d_crc_partial_res, 
        sizeof(uint16_t) * CEIL(N, BLOCK_SIZE) / M, cudaMemcpyDeviceToHost))
    TM_device_dtoh.stop();
    
    TM_host.start();
    for (size_t i = 0; i < (CEIL(N, BLOCK_SIZE) / M); i++) 
    {
        ((pcrc16_params_t *) params)->h_crc_res_dev ^= h_crc_partial_res_dev[i];
    }
    TM_host.stop();

    d_time->is_initialized      = true;
    d_time->is_task_parallelism = false;
    d_time->htod_time   = TM_device_htod.duration();
    d_time->kernel_time = TM_device_kernel.duration() + TM_host.duration();
    d_time->dtoh_time   = TM_device_dtoh.duration();
}

void pcrc16_parallel_task_parallelism(const constants_t *c, void *params, device_time_t *d_time)
{
    Timer<DEVICE> TM_device;
    Timer<HOST> TM_host;

    const size_t  N = c->N;
    const uint8_t STREAM_DIM = c->STREAM_DIM;
    const size_t  SEG_SIZE   = c->SEG_SIZE;

    uint16_t *d_message = ((pcrc16_params_t *) params)->d_message;
    uint16_t *d_beta    = ((pcrc16_params_t *) params)->d_beta; 
    uint16_t *d_crc_partial_res = ((pcrc16_params_t *) params)->d_crc_partial_res;

    uint16_t *h_message = ((pcrc16_params_t *) params)->h_message_16;
    uint16_t *h_beta    = ((pcrc16_params_t *) params)->h_beta;
    uint16_t generator = ((pcrc16_params_t *) params)->generator;
    uint16_t *h_crc_partial_res_dev = 
        ((pcrc16_params_t *) params)->h_crc_partial_res_dev;

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
                    d_message + d_input_offset / M, 
                    h_message + h_input_offset / M,  
                    sizeof(uint16_t) * SEG_SIZE / M, 
                    cudaMemcpyHostToDevice, 
                    stream[stream_index]) 
            )
            SAFE_CALL( 
                cudaMemcpyAsync(
                    d_beta + d_input_offset / M, 
                    h_beta + h_input_offset / M, 
                    sizeof(uint16_t) * SEG_SIZE / M, 
                    cudaMemcpyHostToDevice, 
                    stream[stream_index]) 
            )
        }

        // 2. Call kernels for every streams.
        for (int stream_index = 0; stream_index < STREAM_DIM; stream_index++) {
            int d_output_offset = stream_index * CEIL(SEG_SIZE, BLOCK_SIZE);
            int d_input_offset = stream_index * SEG_SIZE;
            pcrc16_kernel_reduction<<< SEG_SIZE / BLOCK_SIZE, BLOCK_SIZE, 0, stream[stream_index]>>>
                (d_message + d_input_offset / M, d_beta + d_input_offset / M, (uint32_t) generator + 0x10000, d_crc_partial_res + d_output_offset / M);
        }

        // 3. Copy outputs for every streams.
        for (int stream_index = 0; stream_index < STREAM_DIM; stream_index++) {
            int h_output_offset = ((i / SEG_SIZE) + stream_index) * CEIL(SEG_SIZE, BLOCK_SIZE);
            int d_output_offset = stream_index * CEIL(SEG_SIZE, BLOCK_SIZE);
            SAFE_CALL( 
                cudaMemcpyAsync( 
                    h_crc_partial_res_dev + h_output_offset / M, 
                    d_crc_partial_res + d_output_offset / M, 
                    sizeof(uint16_t) * CEIL(SEG_SIZE, BLOCK_SIZE) / M, 
                    cudaMemcpyDeviceToHost,
                    stream[stream_index]) )
        }
    }
    TM_device.stop();

    TM_host.start();
    for (size_t i = 0; i < (CEIL(N, BLOCK_SIZE) / M); i++) 
    {
        ((pcrc16_params_t *) params)->h_crc_res_dev ^= h_crc_partial_res_dev[i];
    }
    TM_host.stop();

    d_time->is_initialized      = true;
    d_time->is_task_parallelism = true;
    d_time->kernel_time = TM_device.duration() + TM_host.duration();
}

bool pcrc16_compare(const constants_t *c, void *params)
{
    uint16_t h_crc_res_dev  = ((pcrc16_params_t *) params)->h_crc_res_dev;
    uint16_t h_crc_res_host = ((pcrc16_params_t *) params)->h_crc_res_host;

    return h_crc_res_dev == h_crc_res_host;
}

bool pcrc16_compare_reduction(const constants_t *c, void *params)
{
    return pcrc16_compare(c, params);
}

bool pcrc16_compare_task_parallelism(const constants_t *c, void *params)
{
    return pcrc16_compare(c, params);
}

void pcrc16_free_common(void *params)
{
    uint8_t  *h_message    = ((pcrc16_params_t *) params)->h_message;
    uint16_t *h_message_16 = ((pcrc16_params_t *) params)->h_message_16;
    uint16_t *h_beta       = ((pcrc16_params_t *) params)->h_beta;
    uint16_t *h_crc_partial_res_dev = 
        ((pcrc16_params_t *) params)->h_crc_partial_res_dev;

    // Free host.
    delete[] h_message;
    delete[] h_message_16;
    delete[] h_beta;
    delete[] h_crc_partial_res_dev;
}

void pcrc16_free_device(void *params)
{
    uint16_t *d_message = ((pcrc16_params_t *) params)->d_message;
    uint16_t *d_beta    = ((pcrc16_params_t *) params)->d_beta; 
    uint16_t *d_crc_partial_res = ((pcrc16_params_t *) params)->d_crc_partial_res;

    // Free device.
    SAFE_CALL(cudaFree(d_message))
    SAFE_CALL(cudaFree(d_beta))
    SAFE_CALL(cudaFree(d_crc_partial_res))
}

void pcrc16_free_device_reduction(void *params)
{
    pcrc16_free_device(params);
}

void pcrc16_free_device_task_parallelism(void *params)
{
    pcrc16_free_device(params);
}


void pcrc16_free(void *params)
{
    // Free host.
    pcrc16_free_common(params);
    // Free device.
    pcrc16_free_device(params);
    // Free params.
    delete ((pcrc16_params_t *) params);
}

void pcrc16_free_reduction(void *params) 
{
    // Free host.
    pcrc16_free_common(params);
    // Free device.
    pcrc16_free_device_reduction(params);
    // Free params.
    delete ((pcrc16_params_t *) params);
}

void pcrc16_free_task_parallelism(void *params) 
{
    // Free host.
    pcrc16_free_common(params);
    // Free device.
    pcrc16_free_device_task_parallelism(params);
    // Free params.
    delete ((pcrc16_params_t *) params);
}
//------------------------------------------------------------------------------

__global__
static void pcrc16_kernel(const uint16_t* d_message, 
                          const uint16_t* d_beta,
                          const uint32_t d_generator,
                          uint16_t *d_partial_crc)
{
    // __shared__ uint8_t ds_mem_message[BLOCK_SIZE];
    // __shared__ uint8_t ds_mem_beta[BLOCK_SIZE];
    __shared__ uint16_t ds_mem_crc[BLOCK_SIZE];
    uint32_t globalIndex = threadIdx.x + blockDim.x * blockIdx.x;

    // ds_mem_message[threadIdx.x] = d_message[globalIndex];
    // ds_mem_beta[threadIdx.x] = d_beta[globalIndex];
    uint16_t w    = d_message[globalIndex];
    uint16_t beta = d_beta[globalIndex];
    // __syncthreads();

    // Binary modulo 2 multiplication between w and beta.
    uint32_t mul = 0;
    for(uint8_t i = 0; i < (sizeof(uint16_t) * 8); i++)
    {
        if (beta & (1U << i))
        {
            mul ^= (uint32_t) w << i;
        }
    }

    uint32_t ret = 0;

    // Compute division of mul result by polynomial generator value.
    for (uint8_t i = 0; i < sizeof(uint32_t) * 8; i++)
    {
        // Compute subtraction.
        if ((ret & 0x00010000) != 0)
        {
            ret = (uint32_t)(ret ^ d_generator);
        }

        // Shift by 1 all the divident buffer.
        ret = (uint32_t)(ret << 1) 
            | (0x00000001 & (mul >> (sizeof(uint32_t) * 8 - i - 1)));
    }

    // Compute the last subtraction.
    if ((ret & 0x00010000) != 0)
    {
        ret = (uint32_t)(ret ^ d_generator);
    }

    ds_mem_crc[threadIdx.x] = (uint16_t) ret;
    __syncthreads();

    if (threadIdx.x == 0) 
    {
        uint16_t partial_crc = 0;
        for (size_t i = 0; i < BLOCK_SIZE; i++)
        {
            partial_crc ^= ds_mem_crc[i];
        }
        d_partial_crc[blockIdx.x] = partial_crc;
    }
}

__global__
static void pcrc16_kernel_reduction(const uint16_t* d_message, 
                                    const uint16_t* d_beta,
                                    const uint32_t d_generator,
                                    uint16_t *d_partial_crc)
{
    // __shared__ uint8_t ds_mem_message[BLOCK_SIZE];
    // __shared__ uint8_t ds_mem_beta[BLOCK_SIZE];
    __shared__ uint16_t ds_mem_crc[BLOCK_SIZE];
    uint32_t globalIndex = threadIdx.x + blockDim.x * blockIdx.x;

    // ds_mem_message[threadIdx.x] = d_message[globalIndex];
    // ds_mem_beta[threadIdx.x] = d_beta[globalIndex];
    uint16_t w    = d_message[globalIndex];
    uint16_t beta = d_beta[globalIndex];
    // __syncthreads();

    // Binary modulo 2 multiplication between w and beta.
    uint32_t mul = 0;
    for(uint8_t i = 0; i < (sizeof(uint16_t) * 8); i++)
    {
        if (beta & (1U << i))
        {
            mul ^= (uint32_t) w << i;
        }
    }

    uint32_t ret = 0;

    // Compute division of mul result by polynomial generator value.
    for (uint8_t i = 0; i < sizeof(uint32_t) * 8; i++)
    {
        // Compute subtraction.
        if ((ret & 0x00010000) != 0)
        {
            ret = (uint32_t)(ret ^ d_generator);
        }

        // Shift by 1 all the divident buffer.
        ret = (uint32_t)(ret << 1) 
            | (0x00000001 & (mul >> (sizeof(uint32_t) * 8 - i - 1)));
    }

    // Compute the last subtraction.
    if ((ret & 0x00010000) != 0)
    {
        ret = (uint32_t)(ret ^ d_generator);
    }

    ds_mem_crc[threadIdx.x] = (uint16_t) ret;
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
