/**
 * @author Mirco De Marchi
 * @date 23/07/2020
 * @brief Header of test CRC utils.
 */

#ifndef TEST_TEST_UTILS_H_
#define TEST_TEST_UTILS_H_

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
//------------------------------------------------------------------------------

/// @brief Check test function type.
typedef bool (*check_t)();

/// @brief Descriptor for tests.
typedef struct {
    check_t func;       ///< Check function to test if result is correct.
    const char *info;   ///< String that describe the test that func do.
} check_descriptor_t;
//------------------------------------------------------------------------------

/**
 * @brief Print the byte-to-byte buffer content that will be processed for the 
 * algorithm.
 * @param buffer The processed data with 8 bit for element.
 * @param length The length of the processed data.
 */
void log_processed_data_8(const uint8_t *buffer, size_t length);

/**
 * @brief Print the byte-to-byte buffer content that will be processed for the 
 * algorithm.
 * @param buffer The processed data with 16 bit for element.
 * @param length The length of the processed data.
 */
void log_processed_data_16(const uint16_t *buffer, size_t length);

/**
 * @brief Print the byte-to-byte buffer content that will be processed for the 
 * algorithm.
 * @param buffer The processed data with 32 bit for element.
 * @param length The length of the processed data.
 */
void log_processed_data_32(const uint32_t *buffer, size_t length);

/**
 * @brief Print the byte-to-byte buffer content that will be processed for the 
 * algorithm.
 * @param buffer The processed data with 64 bit for element.
 * @param length The length of the processed data.
 */
void log_processed_data_64(const uint64_t *buffer, size_t length);

#endif // TEST_TEST_UTILS_H_
