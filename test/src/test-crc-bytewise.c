/**
 * @author Mirco De Marchi
 * @date 11/09/2020
 * @brief Source of test CRC bytewise.
 * 
 * To generate the check value of the buffer tests strings, take a look here:
 * http://www.sunshine2k.de/coding/javascript/crc/crc_js.html
 * 
 * Edit this test source if you want to integrate new CRC bytewise tests. 
 */

#include "test-crc-bytewise.h"

#include "log.h"
//------------------------------------------------------------------------------

/// CRC8 bytewise test data.
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

// !!! Add here new check CRC bytewise values !!!

//------------------------------------------------------------------------------

/**
 * CRC bytewise correctness check functions.
 * All these functions implements a check for the correctness of a CRC bytewise
 * algorithm and return true if the check result is correct, otherwise they 
 * return false.
 */
/// @{

static bool check_crc8_bytewise();
static bool check_crc16_bytewise();
static bool check_crc32_bytewise();
static bool check_crc64_bytewise();

// !!! Add here new check CRC bytewise functions !!!

/// @}
//------------------------------------------------------------------------------

/**
 * @brief Array of all check function to call to perform the CRC bytewise test.
 * Add here your check CRC bytewise function that you have implemented and that 
 * the main test wrapper will call.
 */
static check_descriptor_t crc_bytewise_tests[] = {
    {&check_crc8_bytewise,     "CRC8 bytewise implementation"},
    {&check_crc16_bytewise,    "CRC16 bytewise implementation"},
    {&check_crc32_bytewise,    "CRC32 bytewise implementation"},
    {&check_crc64_bytewise,    "CRC64 bytewise implementation"},
    // !!! Add here new check CRC bytewise signatures !!!
};
//------------------------------------------------------------------------------

static bool check_crc8_bytewise()
{
    const char buffer_test[] = CRC8_BUFFER_TEST;
    uint8_t check_crc = CRC8_CHECK_VALUE;
    uint8_t crc8 = crc8_bytewise(buffer_test, sizeof(buffer_test) - 1, 
        crc8_lu);
    log_processed_data_8(buffer_test, sizeof(buffer_test) - 1);
    LOGD("crc8: 0x%x\n", crc8);
    return crc8 == check_crc;
}

static bool check_crc16_bytewise()
{
    const char buffer_test[] = CRC16_BUFFER_TEST;
    uint16_t check_crc = CRC16_CHECK_VALUE;
    uint16_t crc16 = crc16_bytewise(buffer_test, sizeof(buffer_test) - 1,
        crc16_lu);
    log_processed_data_8(buffer_test, sizeof(buffer_test) - 1);
    LOGD("crc16: 0x%x\n", crc16);
    return crc16 == check_crc;
}

static bool check_crc32_bytewise()
{
    const char buffer_test[] = CRC32_BUFFER_TEST;
    uint32_t check_crc = CRC32_CHECK_VALUE;
    uint32_t crc32 = crc32_bytewise(buffer_test, sizeof(buffer_test) - 1,
        crc32_lu);
    log_processed_data_8(buffer_test, sizeof(buffer_test) - 1);
    LOGD("crc32: 0x%x\n", crc32);
    return crc32 == check_crc;
}

static bool check_crc64_bytewise()
{
    const char buffer_test[] = CRC64_BUFFER_TEST;
    uint64_t check_crc = CRC64_CHECK_VALUE;
    uint64_t crc64 = crc64_bytewise(buffer_test, sizeof(buffer_test) - 1,
        crc64_lu);
    log_processed_data_8(buffer_test, sizeof(buffer_test) - 1);
    LOGD("crc64: 0x%lx\n", crc64);
    return crc64 == check_crc;
}
//------------------------------------------------------------------------------

void test_crc_bytewise()
{
    size_t numof_tests 
        = sizeof(crc_bytewise_tests) / sizeof(check_descriptor_t);
    for (size_t i = 0; i < numof_tests; i++)
    {
        LOGI("-- %s\n", crc_bytewise_tests[i].info);
        if (crc_bytewise_tests[i].func()) 
        {
            LOGI("---- Check OK\n");
        }
        else 
        {
            LOGE("---- Check ERROR\n");
        }
    }
}
