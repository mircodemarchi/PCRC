/**
 * @author Mirco De Marchi
 * @date 23/07/2020
 * @brief Header of test CRC bitwise.
 */

#ifndef TEST_TEST_CRC_BITWISE_H_
#define TEST_TEST_CRC_BITWISE_H_

#include <stdlib.h>
#include <stdbool.h>

#include "crc8-bitwise.h"
#include "crc16-bitwise.h"
#include "crc32-bitwise.h"
#include "crc64-bitwise.h"

#include "test-utils.h"
//------------------------------------------------------------------------------

/**
 * @brief Wrapper of test CRC bitwise implementations.
 */
void test_crc_bitwise();

#endif // TEST_TEST_CRC_BITWISE_H_
