/**
 * @author Mirco De Marchi
 * @date 23/07/2020
 * @brief Test wrapper.
 * 
 * Edit this file if you have integrated new CRC test source code.
 */

#include <stdio.h>

#include "log.h"

#include "test-crc-bitwise.h"
#include "test-crc-bytewise.h"
#include "test-mod2-arithmetic.h"
// !!! Add here the include of the CRC source code to test !!!

//------------------------------------------------------------------------------

/// @brief CRC source test function type.
typedef void (*test_function_t)();

/// @brief Descriptor for CRC source tests.
typedef struct {
    test_function_t func;
    const char *info;
} test_descriptor_t;
//------------------------------------------------------------------------------

/// @brief Array of functions to test.
static test_descriptor_t pcrc_tests[] = {
    {&test_mod2_arithmetic,     "Modulo 2 arithmetic implementations"}, 
    {&test_crc_bitwise,         "CRC bitwise implementations"}, 
    {&test_crc_bytewise,        "CRC bytewise implementations"}, 
    // !!! Add here new CRC test source signatures !!!
};
//------------------------------------------------------------------------------

int main() 
{
    for (size_t i = 0; i < sizeof(pcrc_tests) / sizeof(test_descriptor_t); i++)
    {
        LOGI("START TEST: %s\n", pcrc_tests[i].info);
        pcrc_tests[i].func();
        LOGI("FINISHED TEST: %s\n\n", pcrc_tests[i].info);
    }
}
