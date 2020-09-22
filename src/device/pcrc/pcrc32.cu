/**
 * @file pcrc32.cu
 * @date 08/09/2020
 * @author Mirco De Marchi
 * @brief Source of 32 bit CRC parallel and sequential algorithms.
 */

#include "pcrc32.cuh"

#include <chrono>
#include <random>
#include <endian.h>

#include "Timer.cuh"
#include "CheckError.cuh"

#include "crc32-bitwise.h"
#include "crc32-bytewise.h"
#include "mod2.h"
#include "mul2.h"
#include "crc-utils.h"

using namespace timer;
//------------------------------------------------------------------------------

#define M 4             ///< Size of CRC result.
//------------------------------------------------------------------------------

/**
 * @brief Device CRC32 kernel executed by each GPU thread.
 * @param d_message     Message from which calculate the CRC value.
 * @param d_beta        Array of beta factor.
 * @param d_generator   Polynomial generator.
 * @param d_crc         Pointer to the result of the CRC value.
 */
__global__
static void pcrc32_kernel(const uint32_t* d_message, 
                          const uint32_t* d_beta,
                          const uint64_t d_generator,
                          uint32_t *d_partial_crc);

__global__
static void pcrc32_kernel_reduction(const uint32_t* d_message, 
                                    const uint32_t* d_beta,
                                    const uint64_t d_generator,
                                    uint32_t *d_partial_crc);             

/**
 * @brief Device CRC32 kernel executed by each GPU thread Intel version.
 * @param d_message     Message from which calculate the CRC value.
 * @param d_beta        Array of beta factor.
 * @param d_generator   Polynomial generator.
 * @param d_crc         Pointer to the result of the CRC value.
 */
__global__
static void pcrc32_intel_kernel(const uint32_t* d_message, 
                          const uint32_t* d_beta,
                          const uint64_t d_generator,
                          uint32_t *d_partial_crc);

__global__
static void pcrc32_intel_kernel_reduction(const uint32_t* d_message, 
                                    const uint32_t* d_beta,
                                    const uint64_t d_generator,
                                    uint32_t *d_partial_crc);  

/**
 * Allocate and initialize all the pcrc32_params_t fields with generic 
 * polynomial generator.
 * @param c Constants for memory allocation and task management.
 * @param gen Polynomial generator.
 * @return The pcrc32_params_t in a generic cast.
 */
static void *_pcrc32_init_common(const constants_t *c, uint32_t gen);                

//------------------------------------------------------------------------------

static void *_pcrc32_init_common(const constants_t *c, uint32_t gen)
{
    const size_t N = c->N;

    // Host allocation.
    pcrc32_params_t *params = new pcrc32_params_t;
    params->h_message      = new uint8_t[N];
    params->h_message_32   = new uint32_t[N/M];
    params->h_beta         = new uint32_t[N/M];
    params->h_crc_partial_res_dev = new uint32_t[CEIL(N, BLOCK_SIZE)/M];

    // Init polynomial generator.
    params->generator = gen;

    // Host initialization.
    params->h_crc_res_dev  = 0x00000000;
    params->h_crc_res_host = 0x00000000;

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<uint8_t> distribution(0x00, 0xFF);

    for (size_t i = 0; i < N; i++) 
    {
        params->h_message[i] = distribution(generator);
    }

    // Convert h_message in h_message_32.
    memcpy(params->h_message_32, params->h_message, N);
    for (size_t i = 0; i < N/M; i++)
    {
        params->h_message_32[i] = be32toh(params->h_message_32[i]);
    }

    // Generate beta array.

    // Slow method.
    // for (size_t i = 0; i < N/M; i++) 
    // {
    //     size_t shift_buffer_length = M * (i + 1);
    //     uint8_t *shift_buffer = new uint8_t[shift_buffer_length + 1]();
    //     shift_buffer[0] = 0x01;
    //     params->h_beta[N/M - i - 1] = (uint32_t) mod2_64(
    //         shift_buffer, shift_buffer_length + 1, 
    //         params->generator + 0x100000000);
    //     delete[] shift_buffer;
    // }

    // Fast method.
    uint8_t *beta0 = new uint8_t[M + 1]();
    uint8_t mul64_arr[8] = {};
    beta0[0] = 0x01;
    params->h_beta[N/M - 1] = (uint32_t) mod2_64(beta0, M + 1, 
                              params->generator + 0x100000000);
    delete[] beta0;
    for (size_t i = 1; i < N/M; i++) 
    {
        uint64_t mul64 = mul2_32(params->h_beta[N/M - i], params->h_beta[N/M - 1]);
        mul64 = htobe64(mul64);
        memcpy(mul64_arr, &mul64, 8);
        params->h_beta[N/M - i - 1] = (uint32_t) mod2_64(
            mul64_arr, 8, 
            params->generator + 0x100000000);
    }

    return (void *) params;
}

void *pcrc32_init_common(const constants_t *c)
{
    return _pcrc32_init_common(c, CRC32);
}

void *pcrc32_intel_init_common(const constants_t *c)
{
    return _pcrc32_init_common(c, CRC32_INTEL);
}

void *pcrc32_init_device(const constants_t *c, void *params)
{
    const size_t N = c->N;
    pcrc32_params_t *pcrc32_params = (pcrc32_params_t *) params;

    // Reset tmp data.
    pcrc32_params->h_crc_res_dev  = 0x00000000;
    pcrc32_params->h_crc_res_host = 0x00000000;

    // Device allocation.
    SAFE_CALL(cudaMalloc(&pcrc32_params->d_message , sizeof(uint32_t) * N/M))
    SAFE_CALL(cudaMalloc(&pcrc32_params->d_beta    , sizeof(uint32_t) * N/M))
    SAFE_CALL(cudaMalloc(&pcrc32_params->d_crc_partial_res, 
                         sizeof(uint32_t) * CEIL(N, BLOCK_SIZE) / M))

    return params;
}

void *pcrc32_init_device_reduction(const constants_t *c, void *params)
{
    return pcrc32_init_device(c, params);
}

void *pcrc32_init_device_task_parallelism(const constants_t *c, void *params)
{
    const uint8_t STREAM_DIM = c->STREAM_DIM;
    const size_t  SEG_SIZE   = c->SEG_SIZE;
    pcrc32_params_t *pcrc32_params = (pcrc32_params_t *) params;

    // Reset tmp data.
    pcrc32_params->h_crc_res_dev  = 0x00000000;
    pcrc32_params->h_crc_res_host = 0x00000000;

    // Device allocation.
    SAFE_CALL(cudaMalloc(&pcrc32_params->d_message, 
              sizeof(uint32_t) * SEG_SIZE * STREAM_DIM / M))
    SAFE_CALL(cudaMalloc(&pcrc32_params->d_beta, 
              sizeof(uint32_t) * SEG_SIZE * STREAM_DIM / M))
    SAFE_CALL(cudaMalloc(&pcrc32_params->d_crc_partial_res, 
              sizeof(uint32_t) * CEIL(SEG_SIZE, BLOCK_SIZE) * STREAM_DIM / M))

    return params;
}

void *pcrc32_intel_init_device(const constants_t *c, void *params)
{
    const size_t N = c->N;
    pcrc32_params_t *pcrc32_params = (pcrc32_params_t *) params;

    // Reset tmp data.
    pcrc32_params->h_crc_res_dev  = 0x00000000;
    pcrc32_params->h_crc_res_host = 0x00000000;

    // Device allocation.
    SAFE_CALL(cudaMalloc(&pcrc32_params->d_message , sizeof(uint32_t) * N/M))
    SAFE_CALL(cudaMalloc(&pcrc32_params->d_beta    , sizeof(uint32_t) * N/M))
    SAFE_CALL(cudaMalloc(&pcrc32_params->d_crc_partial_res, 
                         sizeof(uint32_t) * CEIL(N, BLOCK_SIZE) / M))

    return params;
}

void *pcrc32_intel_init_device_reduction(const constants_t *c, void *params)
{
    return pcrc32_intel_init_device(c, params);
}

void *pcrc32_intel_init_device_task_parallelism(const constants_t *c, void *params)
{
    const uint8_t STREAM_DIM = c->STREAM_DIM;
    const size_t  SEG_SIZE   = c->SEG_SIZE;
    pcrc32_params_t *pcrc32_params = (pcrc32_params_t *) params;

    // Reset tmp data.
    pcrc32_params->h_crc_res_dev  = 0x00000000;
    pcrc32_params->h_crc_res_host = 0x00000000;

    // Device allocation.
    SAFE_CALL(cudaMalloc(&pcrc32_params->d_message, 
              sizeof(uint32_t) * SEG_SIZE * STREAM_DIM / M))
    SAFE_CALL(cudaMalloc(&pcrc32_params->d_beta, 
              sizeof(uint32_t) * SEG_SIZE * STREAM_DIM / M))
    SAFE_CALL(cudaMalloc(&pcrc32_params->d_crc_partial_res, 
              sizeof(uint32_t) * CEIL(SEG_SIZE, BLOCK_SIZE) * STREAM_DIM / M))

    return params;
}

void *pcrc32_init(const constants_t *c)
{
    return pcrc32_init_device(c, pcrc32_init_common(c));
}

void *pcrc32_init_reduction(const constants_t *c)
{
    return pcrc32_init_device_reduction(c, pcrc32_init_common(c));
}

void *pcrc32_init_task_parallelism(const constants_t *c)
{
    return pcrc32_init_device_task_parallelism(c, pcrc32_init_common(c));
}

void pcrc32_sequential(const constants_t *c, void *params, host_time_t *h_time)
{
    Timer<HOST> TM_host;

    const size_t N = c->N;

    uint8_t *message  = ((pcrc32_params_t *) params)->h_message;
    uint32_t generator = ((pcrc32_params_t *) params)->generator;
    // TODO: implement crc32 with generator.

    TM_host.start();
    uint32_t crc = crc32_bitwise(message, N);
    ((pcrc32_params_t *) params)->h_crc_res_host = crc;
    TM_host.stop();

    h_time->is_initialized = true;
    h_time->exec_time = TM_host.duration();
}

void pcrc32_sequential_bytewise(const constants_t *c, void *params, host_time_t *h_time)
{
    Timer<HOST> TM_host;

    const size_t N = c->N;

    uint8_t *message  = ((pcrc32_params_t *) params)->h_message;
    uint32_t generator = ((pcrc32_params_t *) params)->generator;
    // TODO: implement crc32 with generator.
    
    TM_host.start();
    uint32_t crc = crc32_bytewise(message, N, crc32_lu);
    ((pcrc32_params_t *) params)->h_crc_res_host = crc;
    TM_host.stop();

    h_time->is_initialized = true;
    h_time->exec_time = TM_host.duration();
}

void pcrc32_intel_sequential(const constants_t *c, void *params, host_time_t *h_time)
{
    Timer<HOST> TM_host;

    const size_t N = c->N;

    uint8_t *message  = ((pcrc32_params_t *) params)->h_message;
    uint32_t generator = ((pcrc32_params_t *) params)->generator;
    // TODO: implement crc32 with generator.

    TM_host.start();
    uint32_t crc = crc32_intel_bitwise(0x00000000, message, N);
    ((pcrc32_params_t *) params)->h_crc_res_host = crc;
    TM_host.stop();

    h_time->is_initialized = true;
    h_time->exec_time = TM_host.duration();
}

void pcrc32_intel_sequential_hw(const constants_t *c, void *params, host_time_t *h_time)
{
    Timer<HOST> TM_host;

    const size_t N = c->N;

    uint8_t *message  = ((pcrc32_params_t *) params)->h_message;
    uint32_t generator = ((pcrc32_params_t *) params)->generator;
    // TODO: implement crc32 with generator.

    TM_host.start();
    uint32_t crc = crc32_intel_hw_bitwise(0x00000000, message, N);
    ((pcrc32_params_t *) params)->h_crc_res_host = crc;
    TM_host.stop();

    h_time->is_initialized = true;
    h_time->exec_time = TM_host.duration();
}


void pcrc32_parallel(const constants_t *c, void *params, device_time_t *d_time)
{
    Timer<DEVICE> TM_device_kernel, TM_device_htod, TM_device_dtoh;
    Timer<HOST> TM_host;

    const size_t N = c->N;

    uint32_t *d_message = ((pcrc32_params_t *) params)->d_message;
    uint32_t *d_beta    = ((pcrc32_params_t *) params)->d_beta; 
    uint32_t *d_crc_partial_res = ((pcrc32_params_t *) params)->d_crc_partial_res;

    uint32_t *h_message = ((pcrc32_params_t *) params)->h_message_32;
    uint32_t *h_beta    = ((pcrc32_params_t *) params)->h_beta;
    uint32_t generator  = ((pcrc32_params_t *) params)->generator;
    uint32_t *h_crc_partial_res_dev = 
        ((pcrc32_params_t *) params)->h_crc_partial_res_dev;

    // Device copy inputs.
    TM_device_htod.start();
    SAFE_CALL(cudaMemcpy(d_message, h_message, sizeof(uint32_t) * N/M, 
        cudaMemcpyHostToDevice))
    SAFE_CALL(cudaMemcpy(d_beta, h_beta, sizeof(uint32_t) * N/M, 
        cudaMemcpyHostToDevice))
    TM_device_htod.stop();

    TM_device_kernel.start();
    // Device dim.
    dim3 DimGrid((N/M) / BLOCK_SIZE, 1, 1);
    if ((N/M) % BLOCK_SIZE) DimGrid.x++;
    dim3 DimBlock(BLOCK_SIZE, 1, 1);

    // Device kernel call.
    pcrc32_kernel<<< DimGrid, DimBlock >>>(d_message, d_beta, 
        (uint64_t) generator + 0x100000000, d_crc_partial_res);
    CHECK_CUDA_ERROR
    TM_device_kernel.stop();

    // Device copy result.
    TM_device_dtoh.start();
    SAFE_CALL(cudaMemcpy(h_crc_partial_res_dev, d_crc_partial_res, 
        sizeof(uint32_t) * CEIL(N, BLOCK_SIZE) / M, cudaMemcpyDeviceToHost))
    TM_device_dtoh.stop();
    
    TM_host.start();
    for (size_t i = 0; i < (CEIL(N, BLOCK_SIZE) / M); i++)
    {
        ((pcrc32_params_t *) params)->h_crc_res_dev ^= h_crc_partial_res_dev[i];
    }
    TM_host.stop();

    d_time->is_initialized      = true;
    d_time->is_task_parallelism = false;
    d_time->htod_time   = TM_device_htod.duration();
    d_time->kernel_time = TM_device_kernel.duration() + TM_host.duration();
    d_time->dtoh_time   = TM_device_dtoh.duration();
}

void pcrc32_parallel_reduction(const constants_t *c, void *params, device_time_t *d_time)
{
    Timer<DEVICE> TM_device_kernel, TM_device_htod, TM_device_dtoh;
    Timer<HOST> TM_host;

    const size_t N = c->N;

    uint32_t *d_message = ((pcrc32_params_t *) params)->d_message;
    uint32_t *d_beta    = ((pcrc32_params_t *) params)->d_beta; 
    uint32_t *d_crc_partial_res = ((pcrc32_params_t *) params)->d_crc_partial_res;

    uint32_t *h_message = ((pcrc32_params_t *) params)->h_message_32;
    uint32_t *h_beta    = ((pcrc32_params_t *) params)->h_beta;
    uint32_t generator  = ((pcrc32_params_t *) params)->generator;
    uint32_t *h_crc_partial_res_dev = 
        ((pcrc32_params_t *) params)->h_crc_partial_res_dev;

    // Device copy inputs.
    TM_device_htod.start();
    SAFE_CALL(cudaMemcpy(d_message, h_message, sizeof(uint32_t) * N/M, 
        cudaMemcpyHostToDevice))
    SAFE_CALL(cudaMemcpy(d_beta, h_beta, sizeof(uint32_t) * N/M, 
        cudaMemcpyHostToDevice))
    TM_device_htod.stop();

    TM_device_kernel.start();
    // Device dim.
    dim3 DimGrid((N/M) / BLOCK_SIZE, 1, 1);
    if ((N/M) % BLOCK_SIZE) DimGrid.x++;
    dim3 DimBlock(BLOCK_SIZE, 1, 1);

    // Device kernel call.
    pcrc32_kernel_reduction<<< DimGrid, DimBlock >>>(d_message, d_beta, 
        (uint32_t) generator + 0x100000000, d_crc_partial_res);
    CHECK_CUDA_ERROR
    TM_device_kernel.stop();

    // Device copy result.
    TM_device_dtoh.start();
    SAFE_CALL(cudaMemcpy(h_crc_partial_res_dev, d_crc_partial_res, 
        sizeof(uint32_t) * CEIL(N, BLOCK_SIZE) / M, cudaMemcpyDeviceToHost))
    TM_device_dtoh.stop();

    TM_host.start();
    for (size_t i = 0; i < (CEIL(N, BLOCK_SIZE) / M); i++) 
    {
        ((pcrc32_params_t *) params)->h_crc_res_dev ^= h_crc_partial_res_dev[i];
    }
    TM_host.stop();

    d_time->is_initialized      = true;
    d_time->is_task_parallelism = false;
    d_time->htod_time   = TM_device_htod.duration();
    d_time->kernel_time = TM_device_kernel.duration() + TM_host.duration();
    d_time->dtoh_time   = TM_device_dtoh.duration();
}

void pcrc32_parallel_task_parallelism(const constants_t *c, void *params, device_time_t *d_time)
{
    Timer<DEVICE> TM_device;
    Timer<HOST> TM_host;

    const size_t  N = c->N;
    const uint8_t STREAM_DIM = c->STREAM_DIM;
    const size_t  SEG_SIZE   = c->SEG_SIZE;

    uint32_t *d_message = ((pcrc32_params_t *) params)->d_message;
    uint32_t *d_beta    = ((pcrc32_params_t *) params)->d_beta; 
    uint32_t *d_crc_partial_res = ((pcrc32_params_t *) params)->d_crc_partial_res;

    uint32_t *h_message = ((pcrc32_params_t *) params)->h_message_32;
    uint32_t *h_beta    = ((pcrc32_params_t *) params)->h_beta;
    uint32_t generator = ((pcrc32_params_t *) params)->generator;
    uint32_t *h_crc_partial_res_dev = 
        ((pcrc32_params_t *) params)->h_crc_partial_res_dev;

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
                    sizeof(uint32_t) * SEG_SIZE / M, 
                    cudaMemcpyHostToDevice, 
                    stream[stream_index]) 
            )
            SAFE_CALL( 
                cudaMemcpyAsync(
                    d_beta + d_input_offset / M, 
                    h_beta + h_input_offset / M, 
                    sizeof(uint32_t) * SEG_SIZE / M, 
                    cudaMemcpyHostToDevice, 
                    stream[stream_index]) 
            )
        }

        // 2. Call kernels for every streams.
        for (int stream_index = 0; stream_index < STREAM_DIM; stream_index++) {
            int d_output_offset = stream_index * CEIL(SEG_SIZE, BLOCK_SIZE);
            int d_input_offset = stream_index * SEG_SIZE;
            pcrc32_kernel_reduction<<< SEG_SIZE / BLOCK_SIZE, BLOCK_SIZE, 0, stream[stream_index]>>>
                (d_message + d_input_offset / M, d_beta + d_input_offset / M, (uint32_t) generator + 0x100000000, d_crc_partial_res + d_output_offset / M);
        }

        // 3. Copy outputs for every streams.
        for (int stream_index = 0; stream_index < STREAM_DIM; stream_index++) {
            int h_output_offset = ((i / SEG_SIZE) + stream_index) * CEIL(SEG_SIZE, BLOCK_SIZE);
            int d_output_offset = stream_index * CEIL(SEG_SIZE, BLOCK_SIZE);
            SAFE_CALL( 
                cudaMemcpyAsync( 
                    h_crc_partial_res_dev + h_output_offset / M, 
                    d_crc_partial_res + d_output_offset / M, 
                    sizeof(uint32_t) * CEIL(SEG_SIZE, BLOCK_SIZE) / M, 
                    cudaMemcpyDeviceToHost,
                    stream[stream_index]) )
        }
    }
    TM_device.stop();

    TM_host.start();
    for (size_t i = 0; i < (CEIL(N, BLOCK_SIZE) / M); i++) 
    {
        ((pcrc32_params_t *) params)->h_crc_res_dev ^= h_crc_partial_res_dev[i];
    }
    TM_host.stop();

    d_time->is_initialized      = true;
    d_time->is_task_parallelism = true;
    d_time->kernel_time = TM_device.duration() + TM_host.duration();
}

void pcrc32_intel_parallel(const constants_t *c, void *params, device_time_t *d_time)
{
    Timer<DEVICE> TM_device_kernel, TM_device_htod, TM_device_dtoh;
    Timer<HOST> TM_host;

    const size_t N = c->N;

    uint32_t *d_message = ((pcrc32_params_t *) params)->d_message;
    uint32_t *d_beta    = ((pcrc32_params_t *) params)->d_beta; 
    uint32_t *d_crc_partial_res = ((pcrc32_params_t *) params)->d_crc_partial_res;

    uint32_t *h_message = ((pcrc32_params_t *) params)->h_message_32;
    uint32_t *h_beta    = ((pcrc32_params_t *) params)->h_beta;
    uint32_t generator  = ((pcrc32_params_t *) params)->generator;
    uint32_t *h_crc_partial_res_dev = 
        ((pcrc32_params_t *) params)->h_crc_partial_res_dev;

    // Device copy inputs.
    TM_device_htod.start();
    SAFE_CALL(cudaMemcpy(d_message, h_message, sizeof(uint32_t) * N/M, 
        cudaMemcpyHostToDevice))
    SAFE_CALL(cudaMemcpy(d_beta, h_beta, sizeof(uint32_t) * N/M, 
        cudaMemcpyHostToDevice))
    TM_device_htod.stop();

    TM_device_kernel.start();
    // Device dim.
    dim3 DimGrid((N/M) / BLOCK_SIZE, 1, 1);
    if ((N/M) % BLOCK_SIZE) DimGrid.x++;
    dim3 DimBlock(BLOCK_SIZE, 1, 1);

    // Device kernel call.
    pcrc32_intel_kernel<<< DimGrid, DimBlock >>>(d_message, d_beta, 
        (uint64_t) generator + 0x100000000, d_crc_partial_res);
    CHECK_CUDA_ERROR
    TM_device_kernel.stop();

    // Device copy result.
    TM_device_dtoh.start();
    SAFE_CALL(cudaMemcpy(h_crc_partial_res_dev, d_crc_partial_res, 
        sizeof(uint32_t) * CEIL(N, BLOCK_SIZE) / M, cudaMemcpyDeviceToHost))
    TM_device_dtoh.stop();

    TM_host.start();
    for (size_t i = 0; i < (CEIL(N, BLOCK_SIZE) / M); i++)
    {
        ((pcrc32_params_t *) params)->h_crc_res_dev ^= h_crc_partial_res_dev[i];
    }

    ((pcrc32_params_t *) params)->h_crc_res_dev 
        = reverse32(((pcrc32_params_t *) params)->h_crc_res_dev);
    TM_host.stop();

    d_time->is_initialized      = true;
    d_time->is_task_parallelism = false;
    d_time->htod_time   = TM_device_htod.duration();
    d_time->kernel_time = TM_device_kernel.duration() + TM_host.duration();
    d_time->dtoh_time   = TM_device_dtoh.duration();
}

void pcrc32_intel_parallel_reduction(const constants_t *c, void *params, device_time_t *d_time)
{
    Timer<DEVICE> TM_device_kernel, TM_device_htod, TM_device_dtoh;
    Timer<HOST> TM_host;

    const size_t N = c->N;

    uint32_t *d_message = ((pcrc32_params_t *) params)->d_message;
    uint32_t *d_beta    = ((pcrc32_params_t *) params)->d_beta; 
    uint32_t *d_crc_partial_res = ((pcrc32_params_t *) params)->d_crc_partial_res;

    uint32_t *h_message = ((pcrc32_params_t *) params)->h_message_32;
    uint32_t *h_beta    = ((pcrc32_params_t *) params)->h_beta;
    uint32_t generator  = ((pcrc32_params_t *) params)->generator;
    uint32_t *h_crc_partial_res_dev = 
        ((pcrc32_params_t *) params)->h_crc_partial_res_dev;

    // Device copy inputs.
    TM_device_htod.start();
    SAFE_CALL(cudaMemcpy(d_message, h_message, sizeof(uint32_t) * N/M, 
        cudaMemcpyHostToDevice))
    SAFE_CALL(cudaMemcpy(d_beta, h_beta, sizeof(uint32_t) * N/M, 
        cudaMemcpyHostToDevice))
    TM_device_htod.stop();

    TM_device_kernel.start();
    // Device dim.
    dim3 DimGrid((N/M) / BLOCK_SIZE, 1, 1);
    if ((N/M) % BLOCK_SIZE) DimGrid.x++;
    dim3 DimBlock(BLOCK_SIZE, 1, 1);

    // Device kernel call.
    pcrc32_intel_kernel_reduction<<< DimGrid, DimBlock >>>(d_message, d_beta, 
        (uint32_t) generator + 0x100000000, d_crc_partial_res);
    CHECK_CUDA_ERROR
    TM_device_kernel.stop();

    // Device copy result.
    TM_device_dtoh.start();
    SAFE_CALL(cudaMemcpy(h_crc_partial_res_dev, d_crc_partial_res, 
        sizeof(uint32_t) * CEIL(N, BLOCK_SIZE) / M, cudaMemcpyDeviceToHost))
    TM_device_dtoh.stop();

    TM_host.start();
    for (size_t i = 0; i < (CEIL(N, BLOCK_SIZE) / M); i++) 
    {
        ((pcrc32_params_t *) params)->h_crc_res_dev ^= h_crc_partial_res_dev[i];
    }

    ((pcrc32_params_t *) params)->h_crc_res_dev 
        = reverse32(((pcrc32_params_t *) params)->h_crc_res_dev);
    TM_host.stop();

    d_time->is_initialized      = true;
    d_time->is_task_parallelism = false;
    d_time->htod_time   = TM_device_htod.duration();
    d_time->kernel_time = TM_device_kernel.duration() + TM_host.duration();
    d_time->dtoh_time   = TM_device_dtoh.duration();
}

void pcrc32_intel_parallel_task_parallelism(const constants_t *c, void *params, device_time_t *d_time)
{
    Timer<DEVICE> TM_device;
    Timer<HOST> TM_host;

    const size_t  N = c->N;
    const uint8_t STREAM_DIM = c->STREAM_DIM;
    const size_t  SEG_SIZE   = c->SEG_SIZE;

    uint32_t *d_message = ((pcrc32_params_t *) params)->d_message;
    uint32_t *d_beta    = ((pcrc32_params_t *) params)->d_beta; 
    uint32_t *d_crc_partial_res = ((pcrc32_params_t *) params)->d_crc_partial_res;

    uint32_t *h_message = ((pcrc32_params_t *) params)->h_message_32;
    uint32_t *h_beta    = ((pcrc32_params_t *) params)->h_beta;
    uint32_t generator = ((pcrc32_params_t *) params)->generator;
    uint32_t *h_crc_partial_res_dev = 
        ((pcrc32_params_t *) params)->h_crc_partial_res_dev;

    TM_device.start();
    // TASK PARALLELISM
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
                    sizeof(uint32_t) * SEG_SIZE / M, 
                    cudaMemcpyHostToDevice, 
                    stream[stream_index]) 
            )
            SAFE_CALL( 
                cudaMemcpyAsync(
                    d_beta + d_input_offset / M, 
                    h_beta + h_input_offset / M, 
                    sizeof(uint32_t) * SEG_SIZE / M, 
                    cudaMemcpyHostToDevice, 
                    stream[stream_index]) 
            )
        }

        // 2. Call kernels for every streams.
        for (int stream_index = 0; stream_index < STREAM_DIM; stream_index++) {
            int d_output_offset = stream_index * CEIL(SEG_SIZE, BLOCK_SIZE);
            int d_input_offset = stream_index * SEG_SIZE;
            pcrc32_intel_kernel_reduction<<< SEG_SIZE / BLOCK_SIZE, BLOCK_SIZE, 0, stream[stream_index]>>>
                (d_message + d_input_offset / M, d_beta + d_input_offset / M, (uint32_t) generator + 0x100000000, d_crc_partial_res + d_output_offset / M);
        }

        // 3. Copy outputs for every streams.
        for (int stream_index = 0; stream_index < STREAM_DIM; stream_index++) {
            int h_output_offset = ((i / SEG_SIZE) + stream_index) * CEIL(SEG_SIZE, BLOCK_SIZE);
            int d_output_offset = stream_index * CEIL(SEG_SIZE, BLOCK_SIZE);
            SAFE_CALL( 
                cudaMemcpyAsync( 
                    h_crc_partial_res_dev + h_output_offset / M, 
                    d_crc_partial_res + d_output_offset / M, 
                    sizeof(uint32_t) * CEIL(SEG_SIZE, BLOCK_SIZE) / M, 
                    cudaMemcpyDeviceToHost,
                    stream[stream_index]) )
        }
    }
    TM_device.stop();

    TM_host.start();
    for (size_t i = 0; i < (CEIL(N, BLOCK_SIZE) / M); i++) 
    {
        ((pcrc32_params_t *) params)->h_crc_res_dev ^= h_crc_partial_res_dev[i];
    }

    ((pcrc32_params_t *) params)->h_crc_res_dev 
        = reverse32(((pcrc32_params_t *) params)->h_crc_res_dev);
    TM_host.stop();

    d_time->is_initialized      = true;
    d_time->is_task_parallelism = true;
    d_time->kernel_time = TM_device.duration() + TM_host.duration();
}

bool pcrc32_compare(const constants_t *c, void *params)
{
    uint32_t h_crc_res_dev  = ((pcrc32_params_t *) params)->h_crc_res_dev;
    uint32_t h_crc_res_host = ((pcrc32_params_t *) params)->h_crc_res_host;

    return h_crc_res_dev == h_crc_res_host;
}

bool pcrc32_compare_reduction(const constants_t *c, void *params)
{
    return pcrc32_compare(c, params);
}

bool pcrc32_compare_task_parallelism(const constants_t *c, void *params)
{
    return pcrc32_compare(c, params);
}

void pcrc32_free_common(void *params)
{
    uint8_t  *h_message    = ((pcrc32_params_t *) params)->h_message;
    uint32_t *h_message_16 = ((pcrc32_params_t *) params)->h_message_32;
    uint32_t *h_beta       = ((pcrc32_params_t *) params)->h_beta;
    uint32_t *h_crc_partial_res_dev = 
        ((pcrc32_params_t *) params)->h_crc_partial_res_dev;
    
    // Free host.
    delete[] h_message;
    delete[] h_message_16;
    delete[] h_beta;
    delete[] h_crc_partial_res_dev;
}

void pcrc32_free_device(void *params)
{
    uint32_t *d_message = ((pcrc32_params_t *) params)->d_message;
    uint32_t *d_beta    = ((pcrc32_params_t *) params)->d_beta; 
    uint32_t *d_crc_partial_res = ((pcrc32_params_t *) params)->d_crc_partial_res;

    // Free device.
    SAFE_CALL(cudaFree(d_message))
    SAFE_CALL(cudaFree(d_beta))
    SAFE_CALL(cudaFree(d_crc_partial_res))
}

void pcrc32_free_device_reduction(void *params)
{
    pcrc32_free_device(params);
}

void pcrc32_free_device_task_parallelism(void *params)
{
    pcrc32_free_device(params);
}

void pcrc32_free(void *params)
{
    // Free host.
    pcrc32_free_common(params);
    // Free device.
    pcrc32_free_device(params);
    // Free params.
    delete ((pcrc32_params_t *) params);
}

void pcrc32_free_reduction(void *params) 
{
    // Free host.
    pcrc32_free_common(params);
    // Free device.
    pcrc32_free_device_reduction(params);
    // Free params.
    delete ((pcrc32_params_t *) params);
}

void pcrc32_free_task_parallelism(void *params) 
{
    // Free host.
    pcrc32_free_common(params);
    // Free device.
    pcrc32_free_device_task_parallelism(params);
    // Free params.
    delete ((pcrc32_params_t *) params);
}
//------------------------------------------------------------------------------

__global__
static void pcrc32_kernel(const uint32_t* d_message, 
                          const uint32_t* d_beta,
                          const uint64_t d_generator,
                          uint32_t *d_partial_crc)
{
    // __shared__ uint8_t ds_mem_message[BLOCK_SIZE];
    // __shared__ uint8_t ds_mem_beta[BLOCK_SIZE];
    __shared__ uint32_t ds_mem_crc[BLOCK_SIZE];
    uint32_t globalIndex = threadIdx.x + blockDim.x * blockIdx.x;

    // ds_mem_message[threadIdx.x] = d_message[globalIndex];
    // ds_mem_beta[threadIdx.x] = d_beta[globalIndex];
    uint32_t w    = d_message[globalIndex];
    uint32_t beta = d_beta[globalIndex];
    // __syncthreads();

    // Binary modulo 2 multiplication between w and beta.
    uint64_t mul = 0;
    for(uint8_t i = 0; i < (sizeof(uint32_t) * 8); i++)
    {
        if (beta & (1U << i))
        {
            mul ^= (uint64_t) w << i;
        }
    }

    uint64_t ret = 0;

    // Compute division of mul result by polynomial generator value.
    for (uint8_t i = 0; i < sizeof(uint64_t) * 8; i++)
    {
        // Compute subtraction.
        if ((ret & 0x0000000100000000) != 0)
        {
            ret = (uint64_t)(ret ^ d_generator);
        }

        // Shift by 1 all the divident buffer.
        ret = (uint64_t)(ret << 1) 
            | (0x0000000000000001 & (mul >> (sizeof(uint64_t) * 8 - i - 1)));
    }

    // Compute the last subtraction.
    if ((ret & 0x0000000100000000) != 0)
    {
        ret = (uint64_t)(ret ^ d_generator);
    }

    ds_mem_crc[threadIdx.x] = (uint32_t) ret;
    __syncthreads();

    if (threadIdx.x == 0) 
    {
        uint32_t partial_crc = 0;
        for (size_t i = 0; i < BLOCK_SIZE; i++)
        {
            partial_crc ^= ds_mem_crc[i];
        }
        d_partial_crc[blockIdx.x] = partial_crc;
    }
}

__global__
static void pcrc32_kernel_reduction(const uint32_t* d_message, 
                                    const uint32_t* d_beta,
                                    const uint64_t d_generator,
                                    uint32_t *d_partial_crc)
{
    // __shared__ uint8_t ds_mem_message[BLOCK_SIZE];
    // __shared__ uint8_t ds_mem_beta[BLOCK_SIZE];
    __shared__ uint32_t ds_mem_crc[BLOCK_SIZE];
    uint32_t globalIndex = threadIdx.x + blockDim.x * blockIdx.x;

    // ds_mem_message[threadIdx.x] = d_message[globalIndex];
    // ds_mem_beta[threadIdx.x] = d_beta[globalIndex];
    uint32_t w    = d_message[globalIndex];
    uint32_t beta = d_beta[globalIndex];
    // __syncthreads();

    // Binary modulo 2 multiplication between w and beta.
    uint64_t mul = 0;
    for(uint8_t i = 0; i < (sizeof(uint32_t) * 8); i++)
    {
        if (beta & (1U << i))
        {
            mul ^= (uint64_t) w << i;
        }
    }

    uint64_t ret = 0;

    // Compute division of mul result by polynomial generator value.
    for (uint8_t i = 0; i < sizeof(uint64_t) * 8; i++)
    {
        // Compute subtraction.
        if ((ret & 0x0000000100000000) != 0)
        {
            ret = (uint64_t)(ret ^ d_generator);
        }

        // Shift by 1 all the divident buffer.
        ret = (uint64_t)(ret << 1) 
            | (0x0000000000000001 & (mul >> (sizeof(uint64_t) * 8 - i - 1)));
    }

    // Compute the last subtraction.
    if ((ret & 0x0000000100000000) != 0)
    {
        ret = (uint64_t)(ret ^ d_generator);
    }

    ds_mem_crc[threadIdx.x] = (uint32_t) ret;
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

__global__
static void pcrc32_intel_kernel(const uint32_t* d_message, 
                          const uint32_t* d_beta,
                          const uint64_t d_generator,
                          uint32_t *d_partial_crc)
{
    // __shared__ uint8_t ds_mem_message[BLOCK_SIZE];
    // __shared__ uint8_t ds_mem_beta[BLOCK_SIZE];
    __shared__ uint32_t ds_mem_crc[BLOCK_SIZE];
    uint32_t globalIndex = threadIdx.x + blockDim.x * blockIdx.x;

    // ds_mem_message[threadIdx.x] = d_message[globalIndex];
    // ds_mem_beta[threadIdx.x] = d_beta[globalIndex];
    uint32_t w    = d_message[globalIndex];
    uint32_t beta = d_beta[globalIndex];
    // __syncthreads();

    // b = (b & 0xF0) >> 4 | (b & 0x0F) << 4; // 0xF0=0b11110000 - 0x0F=0b00001111.
    // b = (b & 0xCC) >> 2 | (b & 0x33) << 2; // 0xCC=0b11001100 - 0x33=0b00110011.
    // b = (b & 0xAA) >> 1 | (b & 0x55) << 1; // 0xAA=0b10101010 - 0x55=0b01010101.

    w = (w & 0xF0F0F0F0) >> 4 | (w & 0x0F0F0F0F) << 4;
    w = (w & 0xCCCCCCCC) >> 2 | (w & 0x33333333) << 2;
    w = (w & 0xAAAAAAAA) >> 1 | (w & 0x55555555) << 1;

    // w = (w & 0xFFFF0000) >> 16 | (w & 0x0000FFFF) << 16; 
    // w = (w & 0xFF00FF00) >> 8 | (w & 0x00FF00FF) << 8; 
    // w = (w & 0xF0F0F0F0) >> 4 | (w & 0x0F0F0F0F) << 4; 
    // w = (w & 0xCCCCCCCC) >> 2 | (w & 0x33333333) << 2;
    // w = (w & 0xAAAAAAAA) >> 1 | (w & 0x55555555) << 1;

    // Binary modulo 2 multiplication between w and beta.
    uint64_t mul = 0;
    for(uint8_t i = 0; i < (sizeof(uint32_t) * 8); i++)
    {
        if (beta & (1U << i))
        {
            mul ^= (uint64_t) w << i;
        }
    }

    uint64_t ret = 0;

    // Compute division of mul result by polynomial generator value.
    for (uint8_t i = 0; i < sizeof(uint64_t) * 8; i++)
    {
        // Compute subtraction.
        if ((ret & 0x0000000100000000) != 0)
        {
            ret = (uint64_t)(ret ^ d_generator);
        }

        // Shift by 1 all the divident buffer.
        ret = (uint64_t)(ret << 1) 
            | (0x0000000000000001 & (mul >> (sizeof(uint64_t) * 8 - i - 1)));
    }

    // Compute the last subtraction.
    if ((ret & 0x0000000100000000) != 0)
    {
        ret = (uint64_t)(ret ^ d_generator);
    }

    ds_mem_crc[threadIdx.x] = (uint32_t) ret;
    __syncthreads();

    if (threadIdx.x == 0) 
    {
        uint32_t partial_crc = 0;
        for (size_t i = 0; i < BLOCK_SIZE; i++)
        {
            partial_crc ^= ds_mem_crc[i];
        }
        d_partial_crc[blockIdx.x] = partial_crc;
    }
}

__global__
static void pcrc32_intel_kernel_reduction(const uint32_t* d_message, 
                                    const uint32_t* d_beta,
                                    const uint64_t d_generator,
                                    uint32_t *d_partial_crc)
{
    // __shared__ uint8_t ds_mem_message[BLOCK_SIZE];
    // __shared__ uint8_t ds_mem_beta[BLOCK_SIZE];
    __shared__ uint32_t ds_mem_crc[BLOCK_SIZE];
    uint32_t globalIndex = threadIdx.x + blockDim.x * blockIdx.x;

    // ds_mem_message[threadIdx.x] = d_message[globalIndex];
    // ds_mem_beta[threadIdx.x] = d_beta[globalIndex];
    uint32_t w    = d_message[globalIndex];
    uint32_t beta = d_beta[globalIndex];
    // __syncthreads();

    w = (w & 0xF0F0F0F0) >> 4 | (w & 0x0F0F0F0F) << 4;
    w = (w & 0xCCCCCCCC) >> 2 | (w & 0x33333333) << 2;
    w = (w & 0xAAAAAAAA) >> 1 | (w & 0x55555555) << 1;

    // w = (w & 0xFFFF0000) >> 16 | (w & 0x0000FFFF) << 16; 
    // w = (w & 0xFF00FF00) >> 8 | (w & 0x00FF00FF) << 8; 
    // w = (w & 0xF0F0F0F0) >> 4 | (w & 0x0F0F0F0F) << 4; 
    // w = (w & 0xCCCCCCCC) >> 2 | (w & 0x33333333) << 2;
    // w = (w & 0xAAAAAAAA) >> 1 | (w & 0x55555555) << 1;

    // Binary modulo 2 multiplication between w and beta.
    uint64_t mul = 0;
    for(uint8_t i = 0; i < (sizeof(uint32_t) * 8); i++)
    {
        if (beta & (1U << i))
        {
            mul ^= (uint64_t) w << i;
        }
    }

    uint64_t ret = 0;

    // Compute division of mul result by polynomial generator value.
    for (uint8_t i = 0; i < sizeof(uint64_t) * 8; i++)
    {
        // Compute subtraction.
        if ((ret & 0x0000000100000000) != 0)
        {
            ret = (uint64_t)(ret ^ d_generator);
        }

        // Shift by 1 all the divident buffer.
        ret = (uint64_t)(ret << 1) 
            | (0x0000000000000001 & (mul >> (sizeof(uint64_t) * 8 - i - 1)));
    }

    // Compute the last subtraction.
    if ((ret & 0x0000000100000000) != 0)
    {
        ret = (uint64_t)(ret ^ d_generator);
    }

    ds_mem_crc[threadIdx.x] = (uint32_t) ret;
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