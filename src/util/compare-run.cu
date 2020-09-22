/**
 * @file compare-run.cu
 * @date 26/07/2020
 * @author Luigi Capogrosso, Mirco De Marchi
 * @brief Source compare run functionality.
 * 
 * Compare two different version of the same algorithm, in particular, it 
 * compares an algorithm on a device (a GPU, parallel) and the other on the 
 * host (a CPU, sequential).
 * All of the necessary informations are save in a compare descriptor structure.
 */

#include "compare-run.cuh"
//------------------------------------------------------------------------------

#define B 1
#define K 1024 * B
#define M 1024 * K
#define G 1024 * M
//------------------------------------------------------------------------------

static void print_host_stats(const char *info, const constants_t *consts, host_time_t *h);
static void print_device_stats(const char *info, const constants_t *consts, device_time_t *d);
static void print_speedup(const char *info, const constants_t *consts,  host_time_t *h, device_time_t *d);
//------------------------------------------------------------------------------

void compare_run(compare_descriptor_t *run)
{
    host_time_t   h_time = { .is_initialized = false };
    device_time_t d_time = { .is_initialized = false,
                             .is_task_parallelism = false };

    if (!run)
    {
        LOGE("Compare descriptor NULL pointer");
        return;
    }

    const constants_t *consts = &(run->constants);

    // Init data.
    void *params = run->init(consts);

    // Host execution.
    LOGI("%s: HOST START\n", run->info);
    run->host_algorithm(consts, params, &h_time);
    print_host_stats(run->info, consts, &h_time);

    // Device execution.
    LOGI("%s: DEVICE START\n", run->info);
    run->device_algorithm(consts, params, &d_time);
    print_device_stats(run->info, consts, &d_time);

    // Print speedup.
    print_speedup(run->info, consts, &h_time, &d_time);
    
    // Check correctness.
    bool check = run->compare(consts, params);
    LOGI("%s: CHECK %s\n", run->info, check ? "OK" : "ERROR");

    // Free data.
    run->free(params);
}

void compare_common_run(const constants_t *consts, void *params, 
                        compare_common_descriptor_t *run)
{
    host_time_t   h_time = { .is_initialized = false };
    device_time_t d_time = { .is_initialized = false,
                             .is_task_parallelism = false };

    if (!run)
    {
        LOGE("Compare descriptor NULL pointer");
        return;
    }

    // Init data.
    params = run->init_device(consts, params);

    // Host execution.
    LOGI("%s: HOST START\n", run->info);
    run->host_algorithm(consts, params, &h_time);
    print_host_stats(run->info, consts, &h_time);

    // Device execution.
    LOGI("%s: DEVICE START\n", run->info);
    run->device_algorithm(consts, params, &d_time);
    print_device_stats(run->info, consts, &d_time);

    // Print speedup.
    print_speedup(run->info, consts, &h_time, &d_time);
    
    // Check correctness.
    bool check = run->compare(consts, params);
    LOGI("%s: CHECK %s\n", run->info, check ? "OK" : "ERROR");

    // Free data.
    run->free_device(params);
}

static void print_host_stats(const char *info, const constants_t *consts, host_time_t *h)
{
    if (h->is_initialized) 
    {
        LOGI("%s: HOST DURATION %.3f ms\n", info, h->exec_time);
    }
}

static void print_device_stats(const char *info, const constants_t *consts, device_time_t *d)
{
    if (d->is_initialized)
    {
        if (!d->is_task_parallelism)
        {
            LOGI("%s: DEVICE PCIe UP DURATION %.3f ms\n", info, d->htod_time);
            LOGI("%s: DEVICE PCIe UP RATE %.3f GB/s\n", info, 
                 (consts->N / (d->htod_time / 1000)) / G);
            LOGI("%s: DEVICE KERNEL DURATION %.3f ms\n", info, d->kernel_time);
            LOGI("%s: DEVICE KERNEL RATE %.3f GB/s\n", info, 
                 (consts->N / (d->kernel_time / 1000)) / G);
            LOGI("%s: DEVICE PCIe DOWN DURATION %.3f ms\n", info, d->dtoh_time);
            LOGI("%s: DEVICE PCIe DOWN RATE %.3f GB/s\n", info, 
                 (consts->N / (d->dtoh_time / 1000)) / G);

            LOGI("%s: DEVICE TOTAL DURATION %.3f ms\n", info,  
                 d->htod_time + d->kernel_time + d->dtoh_time);
            LOGI("%s: DEVICE TOTAL RATE %.3f GB/s\n", info, 
                 (consts->N / ((d->htod_time + d->kernel_time + d->dtoh_time) / 1000)) / G);
        }
        else
        {
            LOGI("%s: DEVICE TOTAL DURATION %.3f ms\n", info, 
                 d->kernel_time);
            LOGI("%s: DEVICE TOTAL RATE %.3f GB/s\n", info, 
                 (consts->N / (d->kernel_time / 1000)) / G);
        }
    }
}

static void print_speedup(const char *info, const constants_t *consts, host_time_t *h, device_time_t *d)
{
    if (d->is_initialized && h->is_initialized)
    {
        if (!d->is_task_parallelism)
        {
            LOGI("%s: -- SPEEDUP  %.5fx\n", info, 
                 h->exec_time / (d->htod_time + d->kernel_time + d->dtoh_time));
            LOGI("%s: -- SPEEDUP WITHOUT TRANSFER %.5fx\n", info, 
                 h->exec_time / d->kernel_time);
        }
        else 
        {
            LOGI("%s: -- SPEEDUP  %.5fx\n", info, 
                 h->exec_time / d->kernel_time);
        }
    }
}