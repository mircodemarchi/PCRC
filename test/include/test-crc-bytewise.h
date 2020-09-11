/**
 * @author Mirco De Marchi
 * @date 11/09/2020
 * @brief Header of test CRC bytewise.
 */

#ifndef TEST_TEST_CRC_BYTEWISE_H_
#define TEST_TEST_CRC_BYTEWISE_H_

#include <stdlib.h>
#include <stdbool.h>

#include "crc8-bytewise.h"
#include "crc16-bytewise.h"
#include "crc32-bytewise.h"
#include "crc64-bytewise.h"

#include "test-utils.h"
//------------------------------------------------------------------------------

/**
 * @brief Wrapper of test CRC bytewise implementations.
 */
void test_crc_bytewise();

#endif // TEST_TEST_CRC_BYTEWISE_H_
