/**
 * @author Mirco De Marchi
 * @date 23/07/2020
 * @brief Source of test CRC bitwise.
 * 
 * To generate the check value of the buffer tests strings, take a look here:
 * http://www.sunshine2k.de/coding/javascript/crc/crc_js.html
 * 
 * Edit this test source if you want to integrate new CRC bitwise tests. 
 */

#include "test-crc-bitwise.h"

#include "log.h"
//------------------------------------------------------------------------------

/// CRC8 bitwise test data.
/// @{
#define CRC8_BUFFER_TEST "Ciao siamo i 3Dev!"
#define CRC8_CHECK_VALUE 0x41           ///< Correct CRC of the BUFFER_TEST.
/// @}

/// CRC16 test data.
/// @{
#define CRC16_BUFFER_TEST "Ciao siamo i 3Dev!"
#define CRC16_CHECK_VALUE 0xD9EE        ///< Correct CRC of the BUFFER_TEST.
/// @}

/// CRC32 test data.
/// @{
#define CRC32_BUFFER_TEST "Ciao siamo i 3Dev!"
#define CRC32_CHECK_VALUE 0x288153C1    ///< Correct CRC of the BUFFER_TEST.
/// @}

/// CRC64 test data.
/// @{
#define CRC64_BUFFER_TEST "Ciao siamo i 3Dev!"
#define CRC64_CHECK_VALUE 0x34FFEAD850BAB8FA ///< Correct CRC of the BUFFER_TEST.
/// @}

// !!! Add here new check CRC bitwise values !!!

//------------------------------------------------------------------------------

/**
 * CRC bitwise correctness check functions.
 * All these functions implements a check for the correctness of a CRC bitwise
 * algorithm and return true if the check result is correct, otherwise they 
 * return false.
 */
/// @{

static bool check_crc8_bitwise();
static bool check_crc16_bitwise();
static bool check_crc32_bitwise();
static bool check_crc64_bitwise();

// !!! Add here new check CRC bitwise functions !!!

/// @}
//------------------------------------------------------------------------------

/**
 * @brief Array of all check function to call to perform the CRC bitwise test.
 * Add here your check CRC bitwise function that you have implemented and that 
 * the main test wrapper will call.
 */
static check_descriptor_t crc_bitwise_tests[] = {
    {&check_crc8_bitwise,     "CRC8 bitwise implementation"},
    {&check_crc16_bitwise,    "CRC16 bitwise implementation"},
    {&check_crc32_bitwise,    "CRC32 bitwise implementation"},
    {&check_crc64_bitwise,    "CRC64 bitwise implementation"},
    // !!! Add here new check CRC bitwise signatures !!!
};
//------------------------------------------------------------------------------

static bool check_crc8_bitwise()
{
    const char buffer_test[] = CRC8_BUFFER_TEST;
    uint8_t check_crc = CRC8_CHECK_VALUE;
    uint8_t crc8 = crc8_bitwise(buffer_test, sizeof(buffer_test) - 1);
    log_processed_data_8(buffer_test, sizeof(buffer_test) - 1);
    LOGD("crc8: 0x%x\n", crc8);
    return crc8 == check_crc;
}

static bool check_crc16_bitwise()
{
    const char buffer_test[] = CRC16_BUFFER_TEST;
    uint16_t check_crc = CRC16_CHECK_VALUE;
    uint16_t crc16 = crc16_bitwise(buffer_test, sizeof(buffer_test) - 1);
    log_processed_data_8(buffer_test, sizeof(buffer_test) - 1);
    LOGD("crc16: 0x%x\n", crc16);
    return crc16 == check_crc;
}

static bool check_crc32_bitwise()
{
    const char buffer_test[] = CRC32_BUFFER_TEST;
    uint32_t check_crc = CRC32_CHECK_VALUE;
    uint32_t crc32 = crc32_bitwise(buffer_test, sizeof(buffer_test) - 1);
    log_processed_data_8(buffer_test, sizeof(buffer_test) - 1);
    LOGD("crc32: 0x%x\n", crc32);
    return crc32 == check_crc;
}

static bool check_crc64_bitwise()
{
    const char buffer_test[] = CRC64_BUFFER_TEST;
    uint64_t check_crc = CRC64_CHECK_VALUE;
    uint64_t crc64 = crc64_bitwise(buffer_test, sizeof(buffer_test) - 1);
    log_processed_data_8(buffer_test, sizeof(buffer_test) - 1);
    LOGD("crc64: 0x%lx\n", crc64);
    return crc64 == check_crc;
}
//------------------------------------------------------------------------------

void test_crc_bitwise()
{
    size_t numof_tests 
        = sizeof(crc_bitwise_tests) / sizeof(check_descriptor_t);
    for (size_t i = 0; i < numof_tests; i++)
    {
        LOGI("-- %s\n", crc_bitwise_tests[i].info);
        if (crc_bitwise_tests[i].func()) 
        {
            LOGI("---- Check OK\n");
        }
        else 
        {
            LOGE("---- Check ERROR\n");
        }
    }
}
