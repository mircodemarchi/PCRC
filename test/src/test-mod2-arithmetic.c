/**
 * @author Mirco De Marchi
 * @date 23/07/2020
 * @brief Source of test binary modulo 2 arithmetic operations.
 * 
 * Generate binary modulo 2 remainder check value:
 * https://asecuritysite.com/comms/mod_div?a=10011&b=11
 * 
 * Generate binary modulo 2 multipy check value:
 * https://www.asecuritysite.com/calculators/mod2?a=10101&b=01110&a=10101&b=01110
 * 
 * Edit this test source if you want to integrate new binary modulo 2 
 * arithmetic tests. 
 */

#include "test-mod2-arithmetic.h"

#include "log.h"
//------------------------------------------------------------------------------

/// Modulo 2 remainder at 8 bit operation test data.
/// @{
#define MODULO2_8_REMAINDER_DIVIDENT_TEST { 0x07, 0xCD }  ///< => 1997.
#define MODULO2_8_REMAINDER_DIVISOR_TEST  { 0x1D }        ///< => 29.

/// Correct remainder of the DIVIDENT_TEST and DIVISOR_TEST.
#define MODULO2_8_REMAINDER_CHECK_VALUE 0b00001100
/// @}

/// Modulo 2 remainder at 16 bit operation test data.
/// @{
#define MODULO2_16_REMAINDER_DIVIDENT_TEST { 0x07, 0xCD }  ///< => 1997.
#define MODULO2_16_REMAINDER_DIVISOR_TEST  { 0x011D }      ///< => 285.

/// Correct remainder of the DIVIDENT_TEST and DIVISOR_TEST.
#define MODULO2_16_REMAINDER_CHECK_VALUE 0b10011110

#define MODULO2_16_U16_REMAINDER_DIVIDENT_TEST 0x07CD   ///< => 1997.
#define MODULO2_16_U16_REMAINDER_DIVISOR_TEST  0x011D   ///< => 285.

/// Correct remainder of the DIVIDENT_TEST and DIVISOR_TEST.
#define MODULO2_16_U16_REMAINDER_CHECK_VALUE 0b10011110
/// @}

/// Modulo 2 remainder at 32 bit operation test data.
/// @{
/// => 130877389.
#define MODULO2_32_REMAINDER_DIVIDENT_TEST { 0x07, 0xCD, 0x07, 0xCD } 
/// => 18678045.
#define MODULO2_32_REMAINDER_DIVISOR_TEST  { 0x011D011D }        

/// Correct remainder of the DIVIDENT_TEST and DIVISOR_TEST.
#define MODULO2_32_REMAINDER_CHECK_VALUE 0x009E009E
/// @}

/// Modulo 2 remainder at 64 bit operation test data.
/// @{
/// => 130877389.
#define MODULO2_64_REMAINDER_DIVIDENT_TEST { 0x07, 0xCD, 0x07, 0xCD } 
/// => 18678045.
#define MODULO2_64_REMAINDER_DIVISOR_TEST  { 0x00000000011D011D }       

/// Correct remainder of the DIVIDENT_TEST and DIVISOR_TEST.
#define MODULO2_64_REMAINDER_CHECK_VALUE 0x00000000009E009E
/// @}


/// Modulo 2 multiply at 8 bit operation test data.
/// @{
#define MODULO2_8_MULTIPLY_OP1_TEST     0xAB    ///< => 171.
#define MODULO2_8_MULTIPLY_OP2_TEST     0xCD    ///< => 205.

/// Correct multiply of the OP1 and OP2.
#define MODULO2_8_MULTIPLY_CHECK_VALUE  30751U
/// @}

/// Modulo 2 multiply at 16 bit operation test data.
/// @{
#define MODULO2_16_MULTIPLY_OP1_TEST     0xABCD    ///< => 43981.
#define MODULO2_16_MULTIPLY_OP2_TEST     0xEF12    ///< => 61202.

/// Correct multiply of the OP1 and OP2.
#define MODULO2_16_MULTIPLY_CHECK_VALUE  1818918986U
/// @}

/// Modulo 2 multiply at 32 bit operation test data.
/// @{
#define MODULO2_32_MULTIPLY_OP1_TEST     0x0001ABCD     ///< => 109517.
#define MODULO2_32_MULTIPLY_OP2_TEST     0x0001EF12     ///< => 126738.

/// Correct multiply of the OP1 and OP2.
#define MODULO2_32_MULTIPLY_CHECK_VALUE  4977950794U
/// @}

// !!! Add here new check binary modulo 2 arithmetic values !!!

//------------------------------------------------------------------------------

/**
 * Modulo 2 arithmetic correctness check functions.
 * All these functions implements a check for the correctness of a binary 
 * modulo 2 arithmetic algorithm and return true if the check result is correct, 
 * otherwise they return false.
 */
/// @{

static bool check_mod2_8_remainder();
static bool check_mod2_16_remainder();
static bool check_mod2_16_u16_remainder();
static bool check_mod2_32_remainder();
static bool check_mod2_64_remainder();

static bool check_mul2_8_remainder();
static bool check_mul2_16_remainder();
static bool check_mul2_32_remainder();

// !!! Add here new check binary modulo 2 arithmetic function declarations !!!

/// @}
//------------------------------------------------------------------------------

/**
 * @brief Array of all check function to call to perform the binary modulo 2 
 * arithmetic test.
 * Add here your check binary modulo 2 arithmetic function that you have i
 * mplemented and that the main test wrapper will call.
 */
static check_descriptor_t mod2_arithmetic_tests[] = {
    // Binary modulo 2 remainder.
    {&check_mod2_8_remainder,       "Binary modulo 2 remainder at 8 bit"},
    {&check_mod2_16_remainder,      "Binary modulo 2 remainder at 16 bit"},
    {&check_mod2_16_u16_remainder,  "Binary modulo 2 with 16 bit remainder,"
                                    "16 bit divident"},
    {&check_mod2_32_remainder,      "Binary modulo 2 remainder at 32 bit"},
    {&check_mod2_64_remainder,      "Binary modulo 2 remainder at 64 bit"},
    // Binary modulo 2 multiply.
    {&check_mul2_8_remainder,       "Binary modulo 2 multiply at 8 bit"},
    {&check_mul2_16_remainder,      "Binary modulo 2 multiply at 16 bit"},
    {&check_mul2_32_remainder,      "Binary modulo 2 multiply at 32 bit"},
    // !!! Add here new check modulo 2 arithmetic signatures !!!
};
//------------------------------------------------------------------------------

static bool check_mod2_8_remainder()
{
    const uint8_t divident_test[] = MODULO2_8_REMAINDER_DIVIDENT_TEST;
    const uint8_t divisor_test[]  = MODULO2_8_REMAINDER_DIVISOR_TEST;
    uint8_t check_remainder = MODULO2_8_REMAINDER_CHECK_VALUE;
    uint8_t remainder = mod2_8(divident_test, 
                               sizeof(divident_test), 
                               divisor_test[0]);
    log_processed_data_8(divident_test, sizeof(divident_test));
    log_processed_data_8(divisor_test, sizeof(divisor_test));
    LOGD("mod2 8 bit remainder: 0x%x\n", remainder);
    return remainder == check_remainder;
}

static bool check_mod2_16_remainder()
{
    const uint8_t divident_test[] = MODULO2_16_REMAINDER_DIVIDENT_TEST;
    const uint16_t divisor_test[] = MODULO2_16_REMAINDER_DIVISOR_TEST;
    uint16_t check_remainder = MODULO2_16_REMAINDER_CHECK_VALUE;
    uint16_t remainder = mod2_16(divident_test, 
                               sizeof(divident_test), 
                               divisor_test[0]);
    log_processed_data_8(divident_test, sizeof(divident_test));
    log_processed_data_16(divisor_test, 
        sizeof(divisor_test) / sizeof(uint16_t));
    LOGD("mod2 16 bit remainder: 0x%x\n", remainder);
    return remainder == check_remainder;
}

static bool check_mod2_16_u16_remainder()
{
    const uint16_t divident_test = MODULO2_16_U16_REMAINDER_DIVIDENT_TEST;
    const uint16_t divisor_test  = MODULO2_16_U16_REMAINDER_DIVISOR_TEST;
    uint16_t check_remainder = MODULO2_16_U16_REMAINDER_CHECK_VALUE;
    uint16_t remainder = mod2_16_u16(divident_test, divisor_test);
    log_processed_data_16(&divident_test, 1);
    log_processed_data_16(&divisor_test , 1);
    LOGD("mod2 16 bit remainder with 16 bit divident: 0x%x\n", remainder);
    return remainder == check_remainder;
}

static bool check_mod2_32_remainder()
{
    const uint8_t divident_test[] = MODULO2_32_REMAINDER_DIVIDENT_TEST;
    const uint32_t divisor_test[] = MODULO2_32_REMAINDER_DIVISOR_TEST;
    uint32_t check_remainder = MODULO2_32_REMAINDER_CHECK_VALUE;
    uint32_t remainder = mod2_32(divident_test, 
                               sizeof(divident_test), 
                               divisor_test[0]);
    log_processed_data_8(divident_test, sizeof(divident_test));
    log_processed_data_32(divisor_test, 
        sizeof(divisor_test) / sizeof(uint32_t));
    LOGD("mod2 32 bit remainder: 0x%x\n", remainder);
    return remainder == check_remainder;
}

static bool check_mod2_64_remainder()
{
    const uint8_t divident_test[] = MODULO2_64_REMAINDER_DIVIDENT_TEST;
    const uint64_t divisor_test[] = MODULO2_64_REMAINDER_DIVISOR_TEST;
    uint64_t check_remainder = MODULO2_64_REMAINDER_CHECK_VALUE;
    uint64_t remainder = mod2_64(divident_test, 
                               sizeof(divident_test), 
                               divisor_test[0]);
    log_processed_data_8(divident_test, sizeof(divident_test));
    log_processed_data_64(divisor_test, 
        sizeof(divisor_test) / sizeof(uint64_t));
    LOGD("mod2 64 bit remainder: 0x%lx\n", remainder);
    return remainder == check_remainder;
}

static bool check_mul2_8_remainder()
{
    const uint8_t op1_test = MODULO2_8_MULTIPLY_OP1_TEST;
    const uint8_t op2_test = MODULO2_8_MULTIPLY_OP2_TEST;
    uint16_t check_multiply = MODULO2_8_MULTIPLY_CHECK_VALUE;
    uint16_t multiply = mul2_8(op1_test, op2_test);
    log_processed_data_8(&op1_test, 1);
    log_processed_data_8(&op2_test, 1);
    LOGD("mod2 8 bit multiply: 0x%x\n", multiply);
    return multiply == check_multiply;
}

static bool check_mul2_16_remainder()
{
    const uint16_t op1_test = MODULO2_16_MULTIPLY_OP1_TEST;
    const uint16_t op2_test = MODULO2_16_MULTIPLY_OP2_TEST;
    uint32_t check_multiply = MODULO2_16_MULTIPLY_CHECK_VALUE;
    uint32_t multiply = mul2_16(op1_test, op2_test);
    log_processed_data_16(&op1_test, 1);
    log_processed_data_16(&op2_test, 1);
    LOGD("mod2 16 bit multiply: 0x%x\n", multiply);
    return multiply == check_multiply;
}

static bool check_mul2_32_remainder()
{
    const uint32_t op1_test = MODULO2_32_MULTIPLY_OP1_TEST;
    const uint32_t op2_test = MODULO2_32_MULTIPLY_OP2_TEST;
    uint64_t check_multiply = MODULO2_32_MULTIPLY_CHECK_VALUE;
    uint64_t multiply = mul2_32(op1_test, op2_test);
    log_processed_data_32(&op1_test, 1);
    log_processed_data_32(&op2_test, 1);
    LOGD("mod2 32 bit multiply: 0x%lx\n", multiply);
    return multiply == check_multiply;
}

//------------------------------------------------------------------------------

void test_mod2_arithmetic()
{
    size_t numof_tests 
        = sizeof(mod2_arithmetic_tests) / sizeof(check_descriptor_t);
    for (size_t i = 0; i < numof_tests; i++)
    {
        LOGI("-- %s\n", mod2_arithmetic_tests[i].info);
        if (mod2_arithmetic_tests[i].func()) 
        {
            LOGI("---- Check OK\n");
        }
        else 
        {
            LOGE("---- Check ERROR\n");
        }
    }
}
