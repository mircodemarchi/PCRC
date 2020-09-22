/**
 * @file main.cu
 * @date 26/07/2020
 * @author Luigi Capogrosso, Mirco De Marchi
 * @brief Start point of the application.
 */

#include <iostream>
#include <arpa/inet.h>

#include "Timer.cuh"
#include "CheckError.cuh"
#include "compare-run.cuh"
#include "log.h"

#include "matrix-multiplication.cuh"
#include "pcrc8.cuh"
#include "pcrc16.cuh"
#include "pcrc32.cuh"
#include "crc8-bytewise.h"
#include "crc16-bytewise.h"
#include "crc32-bytewise.h"
#include "crc64-bytewise.h"
#include "crc-utils.h"
//------------------------------------------------------------------------------

#define TEST_CRC8   1
#define TEST_CRC16  1
#define TEST_CRC32  1

//------------------------------------------------------------------------------

/**
 * @brief Test some execution of PCRC contained in the global array below 
 * called "run".
 */
static void test_pcrc();

/**
 * @brief Test some execution of PCRC8 contained in the global array below 
 * called "run_pcrc8", with the same constants values.
 * @param c Constants value.
 */
static void test_pcrc8(const constants_t *c);
static void test_params_pcrc8(const constants_t *c, void *params);

/**
 * @brief Test some execution of PCRC8 contained in the global array below 
 * called "run_pcrc16", with the same constants values.
 * @param c Constants value.
 */
static void test_pcrc16(const constants_t *c);
static void test_params_pcrc16(const constants_t *c, void *params);

/**
 * @brief Test some execution of PCRC8 contained in the global array below 
 * called "run_pcrc32", with the same constants values.
 * @param c Constants value.
 */
static void test_pcrc32(const constants_t *c);
static void test_params_pcrc32(const constants_t *c, void *params);
static void test_pcrc32_intel(const constants_t *c);
static void test_params_pcrc32_intel(const constants_t *c, void *params);
//------------------------------------------------------------------------------


compare_descriptor_t runs[] = {

    { // MATRIX MULTIPLICATION without shared memory (example).
        .constants          = {.N = 300},
        .init               = &matrix_mul_init,
        .host_algorithm     = &matrix_mul_sequential,
        .device_algorithm   = &matrix_mul_parallel,
        .free               = &matrix_mul_free,
        .compare            = &matrix_mul_compare,
        .info               = "Matrix Multiplication without shared memory"
    },

    { // PCRC 8 bit.
        .constants          = {.N = 1 << 12},
        .init               = &pcrc8_init,
        .host_algorithm     = &pcrc8_sequential,
        .device_algorithm   = &pcrc8_parallel,
        .free               = &pcrc8_free,
        .compare            = &pcrc8_compare,
        .info               = "Parallel CRC 8 bit"
    },

    { // PCRC 8 bit with reduction.
        .constants          = {.N = 1 << 12},
        .init               = &pcrc8_init_reduction,
        .host_algorithm     = &pcrc8_sequential,
        .device_algorithm   = &pcrc8_parallel_reduction,
        .free               = &pcrc8_free_reduction,
        .compare            = &pcrc8_compare_reduction,
        .info               = "Parallel CRC 8 bit with reduction"
    },

    { // PCRC 8 bit with task parallelism.
        .constants          = {.N = 1 << 12, .STREAM_DIM=2, .SEG_SIZE=(1 << 12) / 2},
        .init               = &pcrc8_init_task_parallelism,
        .host_algorithm     = &pcrc8_sequential,
        .device_algorithm   = &pcrc8_parallel_task_parallelism,
        .free               = &pcrc8_free_task_parallelism,
        .compare            = &pcrc8_compare_task_parallelism,
        .info               = "Parallel CRC 8 bit with task parallelism"
    },

    { // PCRC 16 bit.
        .constants          = {.N = 1 << 12},
        .init               = &pcrc16_init,
        .host_algorithm     = &pcrc16_sequential,
        .device_algorithm   = &pcrc16_parallel,
        .free               = &pcrc16_free,
        .compare            = &pcrc16_compare,
        .info               = "Parallel CRC 16 bit"
    },

    { // PCRC 16 bit with reduction.
        .constants          = {.N = 1 << 12},
        .init               = &pcrc16_init_reduction,
        .host_algorithm     = &pcrc16_sequential,
        .device_algorithm   = &pcrc16_parallel_reduction,
        .free               = &pcrc16_free_reduction,
        .compare            = &pcrc16_compare_reduction,
        .info               = "Parallel CRC 16 bit with reduction"
    },

    { // PCRC 16 bit with task parallelism.
        .constants          = {.N = 1 << 12, .STREAM_DIM=2, .SEG_SIZE=(1 << 12) / 2},
        .init               = &pcrc16_init_task_parallelism,
        .host_algorithm     = &pcrc16_sequential,
        .device_algorithm   = &pcrc16_parallel_task_parallelism,
        .free               = &pcrc16_free_task_parallelism,
        .compare            = &pcrc16_compare_task_parallelism,
        .info               = "Parallel CRC 16 bit with task parallelism"
    },

    { // PCRC 32 bit.
        .constants          = {.N = 1 << 12},
        .init               = &pcrc32_init,
        .host_algorithm     = &pcrc32_sequential,
        .device_algorithm   = &pcrc32_parallel,
        .free               = &pcrc32_free,
        .compare            = &pcrc32_compare,
        .info               = "Parallel CRC 32 bit"
    },

    { // PCRC 32 bit with reduction.
        .constants          = {.N = 1 << 12},
        .init               = &pcrc32_init_reduction,
        .host_algorithm     = &pcrc32_sequential,
        .device_algorithm   = &pcrc32_parallel_reduction,
        .free               = &pcrc32_free_reduction,
        .compare            = &pcrc32_compare_reduction,
        .info               = "Parallel CRC 32 bit with reduction"
    },

    { // PCRC 32 bit with task parallelism.
        .constants          = {.N = 1 << 12, .STREAM_DIM=2, .SEG_SIZE=(1 << 12) / 2},
        .init               = &pcrc32_init_task_parallelism,
        .host_algorithm     = &pcrc32_sequential,
        .device_algorithm   = &pcrc32_parallel_task_parallelism,
        .free               = &pcrc32_free_task_parallelism,
        .compare            = &pcrc32_compare_task_parallelism,
        .info               = "Parallel CRC 32 bit with task parallelism"
    },

    // !!! Insert here all the compare descriptor data to run a compare !!!

};

// ======== TEST PCRC8 ========
compare_common_descriptor_t runs_pcrc8[] = {
    { // PCRC 8 bit.
        .init_device        = &pcrc8_init_device,
        .host_algorithm     = &pcrc8_sequential,
        .device_algorithm   = &pcrc8_parallel,
        .free_device        = &pcrc8_free_device,
        .compare            = &pcrc8_compare,
        .info               = "Parallel CRC 8 bit bitwise"
    },
    { // PCRC 8 bit with reduction.
        .init_device        = &pcrc8_init_device_reduction,
        .host_algorithm     = &pcrc8_sequential,
        .device_algorithm   = &pcrc8_parallel_reduction,
        .free_device        = &pcrc8_free_device_reduction,
        .compare            = &pcrc8_compare_reduction,
        .info               = "Parallel CRC 8 bit bitwise with reduction"
    },
    { // PCRC 8 bit with task parallelism.
        .init_device        = &pcrc8_init_device_task_parallelism,
        .host_algorithm     = &pcrc8_sequential,
        .device_algorithm   = &pcrc8_parallel_task_parallelism,
        .free_device        = &pcrc8_free_device_task_parallelism,
        .compare            = &pcrc8_compare_task_parallelism,
        .info               = "Parallel CRC 8 bit bitwise with task parallelism"
    },
    { // PCRC 8 bit bytewise comparison.
        .init_device        = &pcrc8_init_device,
        .host_algorithm     = &pcrc8_sequential_bytewise,
        .device_algorithm   = &pcrc8_parallel,
        .free_device        = &pcrc8_free_device,
        .compare            = &pcrc8_compare,
        .info               = "Parallel CRC 8 bit bytewise"
    },
    { // PCRC 8 bit bytewise comparison with reduction.
        .init_device        = &pcrc8_init_device_reduction,
        .host_algorithm     = &pcrc8_sequential_bytewise,
        .device_algorithm   = &pcrc8_parallel_reduction,
        .free_device        = &pcrc8_free_device_reduction,
        .compare            = &pcrc8_compare_reduction,
        .info               = "Parallel CRC 8 bit bytewise with reduction"
    },
    { // PCRC 8 bit bytewise comparison with task parallelism.
        .init_device        = &pcrc8_init_device_task_parallelism,
        .host_algorithm     = &pcrc8_sequential_bytewise,
        .device_algorithm   = &pcrc8_parallel_task_parallelism,
        .free_device        = &pcrc8_free_device_task_parallelism,
        .compare            = &pcrc8_compare_task_parallelism,
        .info               = "Parallel CRC 8 bit bytewise with task parallelism"
    },
};
// ======== TEST PCRC16 ========
compare_common_descriptor_t runs_pcrc16[] = {
    { // PCRC 16 bit.
        .init_device        = &pcrc16_init_device,
        .host_algorithm     = &pcrc16_sequential,
        .device_algorithm   = &pcrc16_parallel,
        .free_device        = &pcrc16_free_device,
        .compare            = &pcrc16_compare,
        .info               = "Parallel CRC 16 bit bitwise"
    },
    { // PCRC 16 bit with reduction.
        .init_device        = &pcrc16_init_device_reduction,
        .host_algorithm     = &pcrc16_sequential,
        .device_algorithm   = &pcrc16_parallel_reduction,
        .free_device        = &pcrc16_free_device_reduction,
        .compare            = &pcrc16_compare_reduction,
        .info               = "Parallel CRC 16 bit bitwise with reduction"
    },
    { // PCRC 16 bit with task parallelism.
        .init_device        = &pcrc16_init_device_task_parallelism,
        .host_algorithm     = &pcrc16_sequential,
        .device_algorithm   = &pcrc16_parallel_task_parallelism,
        .free_device        = &pcrc16_free_device_task_parallelism,
        .compare            = &pcrc16_compare_task_parallelism,
        .info               = "Parallel CRC 16 bit bitwise with task parallelism"
    },
    { // PCRC 16 bit bytewise comparison.
        .init_device        = &pcrc16_init_device,
        .host_algorithm     = &pcrc16_sequential_bytewise,
        .device_algorithm   = &pcrc16_parallel,
        .free_device        = &pcrc16_free_device,
        .compare            = &pcrc16_compare,
        .info               = "Parallel CRC 16 bit bytewise"
    },
    { // PCRC 16 bit bytewise comparison with reduction.
        .init_device        = &pcrc16_init_device_reduction,
        .host_algorithm     = &pcrc16_sequential_bytewise,
        .device_algorithm   = &pcrc16_parallel_reduction,
        .free_device        = &pcrc16_free_device_reduction,
        .compare            = &pcrc16_compare_reduction,
        .info               = "Parallel CRC 16 bit bytewise with reduction"
    },
    { // PCRC 16 bit bytewise comparison with task parallelism.
        .init_device        = &pcrc16_init_device_task_parallelism,
        .host_algorithm     = &pcrc16_sequential_bytewise,
        .device_algorithm   = &pcrc16_parallel_task_parallelism,
        .free_device        = &pcrc16_free_device_task_parallelism,
        .compare            = &pcrc16_compare_task_parallelism,
        .info               = "Parallel CRC 16 bit bytewise with task parallelism"
    },
};
// ======== TEST PCRC32 ========
compare_common_descriptor_t runs_pcrc32[] = {
    { // PCRC 32 bit.
        .init_device        = &pcrc32_init_device,
        .host_algorithm     = &pcrc32_sequential,
        .device_algorithm   = &pcrc32_parallel,
        .free_device        = &pcrc32_free_device,
        .compare            = &pcrc32_compare,
        .info               = "Parallel CRC 32 bit standard bitwise"
    },
    { // PCRC 32 bit with reduction.
        .init_device        = &pcrc32_init_device_reduction,
        .host_algorithm     = &pcrc32_sequential,
        .device_algorithm   = &pcrc32_parallel_reduction,
        .free_device        = &pcrc32_free_device_reduction,
        .compare            = &pcrc32_compare_reduction,
        .info               = "Parallel CRC 32 bit standard bitwise with reduction"
    },
    { // PCRC 32 bit with task parallelism.
        .init_device        = &pcrc32_init_device_task_parallelism,
        .host_algorithm     = &pcrc32_sequential,
        .device_algorithm   = &pcrc32_parallel_task_parallelism,
        .free_device        = &pcrc32_free_device_task_parallelism,
        .compare            = &pcrc32_compare_task_parallelism,
        .info               = "Parallel CRC 32 bit standard bitwise with task parallelism"
    },
    { // PCRC 32 bit bytewise comparison.
        .init_device        = &pcrc32_init_device,
        .host_algorithm     = &pcrc32_sequential_bytewise,
        .device_algorithm   = &pcrc32_parallel,
        .free_device        = &pcrc32_free_device,
        .compare            = &pcrc32_compare,
        .info               = "Parallel CRC 32 bit bytewise"
    },
    { // PCRC 32 bit bytewise comparison with reduction.
        .init_device        = &pcrc32_init_device_reduction,
        .host_algorithm     = &pcrc32_sequential_bytewise,
        .device_algorithm   = &pcrc32_parallel_reduction,
        .free_device        = &pcrc32_free_device_reduction,
        .compare            = &pcrc32_compare_reduction,
        .info               = "Parallel CRC 32 bit bytewise with reduction"
    },
    { // PCRC 32 bit bytewise comparison with task parallelism.
        .init_device        = &pcrc32_init_device_task_parallelism,
        .host_algorithm     = &pcrc32_sequential_bytewise,
        .device_algorithm   = &pcrc32_parallel_task_parallelism,
        .free_device        = &pcrc32_free_device_task_parallelism,
        .compare            = &pcrc32_compare_task_parallelism,
        .info               = "Parallel CRC 32 bit bytewise with task parallelism"
    },
};

compare_common_descriptor_t runs_pcrc32_intel[] = {
    { // PCRC 32 bit intel sw comparison.
        .init_device        = &pcrc32_intel_init_device,
        .host_algorithm     = &pcrc32_intel_sequential,
        .device_algorithm   = &pcrc32_intel_parallel,
        .free_device        = &pcrc32_free_device,
        .compare            = &pcrc32_compare,
        .info               = "Parallel CRC 32 bit Intel SW bitwise"
    },
    { // PCRC 32 bit intel sw comparison with reduction.
        .init_device        = &pcrc32_intel_init_device_reduction,
        .host_algorithm     = &pcrc32_intel_sequential,
        .device_algorithm   = &pcrc32_intel_parallel_reduction,
        .free_device        = &pcrc32_free_device_reduction,
        .compare            = &pcrc32_compare_reduction,
        .info               = "Parallel CRC 32 bit Intel SW bitwise with reduction"
    },
    { // PCRC 32 bit intel sw comparison with task parallelism.
        .init_device        = &pcrc32_intel_init_device_task_parallelism,
        .host_algorithm     = &pcrc32_intel_sequential,
        .device_algorithm   = &pcrc32_intel_parallel_task_parallelism,
        .free_device        = &pcrc32_free_device_task_parallelism,
        .compare            = &pcrc32_compare_task_parallelism,
        .info               = "Parallel CRC 32 bit Intel SW bitwise with task parallelism"
    },
    { // PCRC 32 bit intel hw comparison.
        .init_device        = &pcrc32_intel_init_device,
        .host_algorithm     = &pcrc32_intel_sequential_hw,
        .device_algorithm   = &pcrc32_intel_parallel,
        .free_device        = &pcrc32_free_device,
        .compare            = &pcrc32_compare,
        .info               = "Parallel CRC 32 bit Intel HW bitwise"
    },
    { // PCRC 32 bit intel hw comparison with reduction.
        .init_device        = &pcrc32_intel_init_device_reduction,
        .host_algorithm     = &pcrc32_intel_sequential_hw,
        .device_algorithm   = &pcrc32_intel_parallel_reduction,
        .free_device        = &pcrc32_free_device_reduction,
        .compare            = &pcrc32_compare_reduction,
        .info               = "Parallel CRC 32 bit Intel HW bitwise with reduction"
    },
    { // PCRC 32 bit intel hw comparison with task parallelism.
        .init_device        = &pcrc32_intel_init_device_task_parallelism,
        .host_algorithm     = &pcrc32_intel_sequential_hw,
        .device_algorithm   = &pcrc32_intel_parallel_task_parallelism,
        .free_device        = &pcrc32_free_device_task_parallelism,
        .compare            = &pcrc32_compare_task_parallelism,
        .info               = "Parallel CRC 32 bit Intel HW bitwise with task parallelism"
    },
};
//------------------------------------------------------------------------------

static void test_pcrc()
{
    size_t numof_compare = sizeof(runs) / sizeof(compare_descriptor_t);
    for (size_t i = 0; i < numof_compare; i++)
    {
        compare_run(&runs[i]);
    }
}

static void test_pcrc8(const constants_t *c)
{
#if TEST_CRC8
    void *common_params_pcrc8 = pcrc8_init_common(c);
    test_params_pcrc8(c, common_params_pcrc8);
    pcrc8_free_common(common_params_pcrc8);
#endif
}

static void test_params_pcrc8(const constants_t *c, void *params)
{
    size_t numof_compare_pcrc8 = sizeof(runs_pcrc8) 
                               / sizeof(compare_common_descriptor_t);
    for (size_t i = 0; i < numof_compare_pcrc8; i++)
    {
        compare_common_run(c, params, &runs_pcrc8[i]);
    }
}

static void test_pcrc16(const constants_t *c)
{
#if TEST_CRC16
    void *common_params_pcrc16 = pcrc16_init_common(c);
    test_params_pcrc16(c, common_params_pcrc16);
    pcrc16_free_common(common_params_pcrc16);
#endif
}

static void test_params_pcrc16(const constants_t *c, void *params)
{
    size_t numof_compare_pcrc16 = sizeof(runs_pcrc16) 
                                / sizeof(compare_common_descriptor_t);
    for (size_t i = 0; i < numof_compare_pcrc16; i++)
    {
        compare_common_run(c, params, &runs_pcrc16[i]);
    }
}

static void test_pcrc32(const constants_t *c)
{
#if TEST_CRC32
    void *common_params_pcrc32 = pcrc32_init_common(c);
    test_params_pcrc32(c, common_params_pcrc32);
    pcrc32_free_common(common_params_pcrc32);
#endif
}

static void test_params_pcrc32(const constants_t *c, void *params)
{
    size_t numof_compare_pcrc32 = sizeof(runs_pcrc32) 
                                / sizeof(compare_common_descriptor_t);
    for (size_t i = 0; i < numof_compare_pcrc32; i++)
    {
        compare_common_run(c, params, &runs_pcrc32[i]);
    }
}

static void test_pcrc32_intel(const constants_t *c)
{
#if TEST_CRC32
    void *common_params_pcrc32 = pcrc32_intel_init_common(c);
    test_params_pcrc32_intel(c, common_params_pcrc32);
    pcrc32_free_common(common_params_pcrc32);
#endif
}

static void test_params_pcrc32_intel(const constants_t *c, void *params)
{
    size_t numof_compare_pcrc32 = sizeof(runs_pcrc32_intel) 
                                / sizeof(compare_common_descriptor_t);
    for (size_t i = 0; i < numof_compare_pcrc32; i++)
    {
        compare_common_run(c, params, &runs_pcrc32_intel[i]);
    }
}
//------------------------------------------------------------------------------

int main()
{
#if 0
    // test_pcrc();

    const size_t N = 1 << 16;
    const uint8_t STREAM_DIM = 2;
    constants_t constant = {.N = N, .STREAM_DIM=STREAM_DIM, 
        .SEG_SIZE=N/STREAM_DIM};
    LOGI("======= N=2^16; BLOCK_SIZE=%d; STREAM_DIM=%d; SEG_SIZE=N/STREAM_DIM; =======\n", 
         BLOCK_SIZE, constant.STREAM_DIM);
    test_pcrc8(&constant);
    test_pcrc16(&constant);
    test_pcrc32(&constant);
#endif

#if 0
    uint8_t crc8_lookup[256];
    uint16_t crc16_lookup[256];
    uint32_t crc32_lookup[256];
    uint64_t crc64_lookup[256];
    print_crc8_lu(generate_crc8_lu(crc8_lookup));
    print_crc16_lu(generate_crc16_lu(crc16_lookup));
    print_crc32_lu(generate_crc32_lu(crc32_lookup));
    print_crc64_lu(generate_crc64_lu(crc64_lookup));

    uint8_t reverse8_lookup[256];
    print_reverse8_lu(generate_reverse8_lu(reverse8_lookup));
#endif

#if 1
    size_t N;
    uint8_t STREAM_DIM;
    constants_t constant;

    // Test.
    N = 1 << 16;
    STREAM_DIM = 2;
    constant = {.N = N, .STREAM_DIM=STREAM_DIM, .SEG_SIZE=N/STREAM_DIM};
    LOGI("======= N=2^16; BLOCK_SIZE=%d; STREAM_DIM=%d; SEG_SIZE=N/STREAM_DIM; =======\n", 
         BLOCK_SIZE, constant.STREAM_DIM);
    test_pcrc8(&constant);
    test_pcrc16(&constant);
    test_pcrc32(&constant);
    test_pcrc32_intel(&constant);

    N = 1 << 16;
    STREAM_DIM = 4;
    constant = {.N = N, .STREAM_DIM=STREAM_DIM, .SEG_SIZE=N/STREAM_DIM};
    LOGI("======= N=2^16; BLOCK_SIZE=%d; STREAM_DIM=%d; SEG_SIZE=N/STREAM_DIM; =======\n", 
         BLOCK_SIZE, constant.STREAM_DIM);
    test_pcrc8(&constant);
    test_pcrc16(&constant);
    test_pcrc32(&constant);
    test_pcrc32_intel(&constant);

    N = 1 << 16;
    STREAM_DIM = 8;
    constant = {.N = N, .STREAM_DIM=STREAM_DIM, .SEG_SIZE=N/STREAM_DIM};
    LOGI("======= N=2^16; BLOCK_SIZE=%d; STREAM_DIM=%d; SEG_SIZE=N/STREAM_DIM; =======\n", 
         BLOCK_SIZE, constant.STREAM_DIM);
    test_pcrc8(&constant);
    test_pcrc16(&constant);
    test_pcrc32(&constant);
    test_pcrc32_intel(&constant);

    N = 1 << 16;
    STREAM_DIM = 16;
    constant = {.N = N, .STREAM_DIM=STREAM_DIM, .SEG_SIZE=N/STREAM_DIM};
    LOGI("======= N=2^16; BLOCK_SIZE=%d; STREAM_DIM=%d; SEG_SIZE=N/STREAM_DIM; =======\n", 
         BLOCK_SIZE, constant.STREAM_DIM);
    test_pcrc8(&constant);
    test_pcrc16(&constant);
    test_pcrc32(&constant);
    test_pcrc32_intel(&constant);

    N = 1 << 16;
    STREAM_DIM = 32;
    constant = {.N = N, .STREAM_DIM=STREAM_DIM, .SEG_SIZE=N/STREAM_DIM};
    LOGI("======= N=2^16; BLOCK_SIZE=%d; STREAM_DIM=%d; SEG_SIZE=N/STREAM_DIM; =======\n", 
         BLOCK_SIZE, constant.STREAM_DIM);
    test_pcrc8(&constant);
    test_pcrc16(&constant);
    test_pcrc32(&constant);
    test_pcrc32_intel(&constant);

    N = 1 << 16;
    STREAM_DIM = 4;
    constant = {.N = N, .STREAM_DIM=STREAM_DIM, .SEG_SIZE=N/8};
    LOGI("======= N=2^16; BLOCK_SIZE=%d; STREAM_DIM=%d; SEG_SIZE=N/8; =======\n", 
         BLOCK_SIZE, constant.STREAM_DIM);
    test_pcrc8(&constant);
    test_pcrc16(&constant);
    test_pcrc32(&constant);
    test_pcrc32_intel(&constant);

    N = 1 << 16;
    STREAM_DIM = 4;
    constant = {.N = N, .STREAM_DIM=STREAM_DIM, .SEG_SIZE=N/16};
    LOGI("======= N=2^16; BLOCK_SIZE=%d; STREAM_DIM=%d; SEG_SIZE=N/16; =======\n", 
         BLOCK_SIZE, constant.STREAM_DIM);
    test_pcrc8(&constant);
    test_pcrc16(&constant);
    test_pcrc32(&constant);
    test_pcrc32_intel(&constant);

    N = 1 << 16;
    STREAM_DIM = 4;
    constant = {.N = N, .STREAM_DIM=STREAM_DIM, .SEG_SIZE=N/32};
    LOGI("======= N=2^16; BLOCK_SIZE=%d; STREAM_DIM=%d; SEG_SIZE=N/32; =======\n", 
         BLOCK_SIZE, constant.STREAM_DIM);
    test_pcrc8(&constant);
    test_pcrc16(&constant);
    test_pcrc32(&constant);
    test_pcrc32_intel(&constant);

    N = 1 << 10;
    STREAM_DIM = 4;
    constant = {.N = N, .STREAM_DIM=STREAM_DIM, .SEG_SIZE=N/STREAM_DIM};
    LOGI("======= N=2^10; BLOCK_SIZE=%d; STREAM_DIM=%d; SEG_SIZE=N/STREAM_DIM; =======\n", 
         BLOCK_SIZE, constant.STREAM_DIM);
    test_pcrc8(&constant);
    test_pcrc16(&constant);
    test_pcrc32(&constant);
    test_pcrc32_intel(&constant);

    N = 1 << 13;
    STREAM_DIM = 4;
    constant = {.N = N, .STREAM_DIM=STREAM_DIM, .SEG_SIZE=N/STREAM_DIM};
    LOGI("======= N=2^13; BLOCK_SIZE=%d; STREAM_DIM=%d; SEG_SIZE=N/STREAM_DIM; =======\n", 
         BLOCK_SIZE, constant.STREAM_DIM);
    test_pcrc8(&constant);
    test_pcrc16(&constant);
    test_pcrc32(&constant);
    test_pcrc32_intel(&constant);

    N = 1 << 18;
    STREAM_DIM = 4;
    constant = {.N = N, .STREAM_DIM=STREAM_DIM, .SEG_SIZE=N/STREAM_DIM};
    LOGI("======= N=2^18; BLOCK_SIZE=%d; STREAM_DIM=%d; SEG_SIZE=N/STREAM_DIM; =======\n", 
         BLOCK_SIZE, constant.STREAM_DIM);
    test_pcrc8(&constant);
    test_pcrc16(&constant);
    test_pcrc32(&constant);
    test_pcrc32_intel(&constant);

    N = 1 << 20;
    STREAM_DIM = 4;
    constant = {.N = N, .STREAM_DIM=STREAM_DIM, .SEG_SIZE=N/STREAM_DIM};
    LOGI("======= N=2^20; BLOCK_SIZE=%d; STREAM_DIM=%d; SEG_SIZE=N/STREAM_DIM; =======\n", 
         BLOCK_SIZE, constant.STREAM_DIM);
    test_pcrc8(&constant);
    test_pcrc16(&constant);
    test_pcrc32(&constant);
    test_pcrc32_intel(&constant);

    N = 1 << 22;
    STREAM_DIM = 4;
    constant = {.N = N, .STREAM_DIM=STREAM_DIM, .SEG_SIZE=N/STREAM_DIM};
    LOGI("======= N=2^22; BLOCK_SIZE=%d; STREAM_DIM=%d; SEG_SIZE=N/STREAM_DIM; =======\n", 
         BLOCK_SIZE, constant.STREAM_DIM);
    test_pcrc8(&constant);
    test_pcrc16(&constant);
    test_pcrc32(&constant);
    test_pcrc32_intel(&constant);
#endif
}