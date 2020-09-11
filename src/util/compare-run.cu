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

using namespace timer;
//------------------------------------------------------------------------------

void compare_run(compare_descriptor_t *run)
{
    Timer<DEVICE> TM_device;
    Timer<HOST>   TM_host;

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
    TM_host.start();
    run->host_algorithm(consts, params);
    TM_host.stop();
    LOGI("%s: HOST DURATION %.3f ms\n", run->info, TM_host.duration());

    // Device execution.
    LOGI("%s: DEVICE START\n", run->info);
    TM_device.start();
    run->device_algorithm(consts, params);
    TM_device.stop();
    LOGI("%s: DEVICE DURATION %.3f ms\n", run->info, TM_device.duration());

    // Print speedup.
    LOGI("%s: -- SPEEDUP %.5f\n", run->info, 
        TM_host.duration() / TM_device.duration());
    
    // Check correctness.
    bool check = run->compare(consts, params);
    LOGI("%s: CHECK %s\n", run->info, check ? "OK" : "ERROR");

    // Free data.
    run->free(params);
}

void compare_common_run(const constants_t *consts, void *params, 
                        compare_common_descriptor_t *run)
{
    Timer<DEVICE> TM_device;
    Timer<HOST>   TM_host;

    if (!run)
    {
        LOGE("Compare descriptor NULL pointer");
        return;
    }

    // Init data.
    params = run->init_device(consts, params);

    // Host execution.
    LOGI("%s: HOST START\n", run->info);
    TM_host.start();
    run->host_algorithm(consts, params);
    TM_host.stop();
    LOGI("%s: HOST DURATION %.3f ms\n", run->info, TM_host.duration());

    // Device execution.
    LOGI("%s: DEVICE START\n", run->info);
    TM_device.start();
    run->device_algorithm(consts, params);
    TM_device.stop();
    LOGI("%s: DEVICE DURATION %.3f ms\n", run->info, TM_device.duration());

    // Print speedup.
    LOGI("%s: -- SPEEDUP %.5f\n", run->info, 
        TM_host.duration() / TM_device.duration());
    
    // Check correctness.
    bool check = run->compare(consts, params);
    LOGI("%s: CHECK %s\n", run->info, check ? "OK" : "ERROR");

    // Free data.
    run->free_device(params);
}