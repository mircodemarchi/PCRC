/**
 * @author Mirco De Marchi
 * @date 11/09/2020
 * @brief Header file of CRC32 bytewise implementation.
 */

#ifndef CRC_CRC_BYTEWISE_CRC32_BYTEWISE_H_
#define CRC_CRC_BYTEWISE_CRC32_BYTEWISE_H_

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "generator.h"
//------------------------------------------------------------------------------

#ifdef __cplusplus
extern "C" {
#endif

extern const uint32_t crc32_lu[];
//------------------------------------------------------------------------------

/**
 * @brief CRC32 bytewise generic algorithm with lookup table already 
 * initialized.
 * @param buffer Message from which generate the CRC32 code.
 * @param length Size of the input message.
 * @param lu     Lookup table already initialized.
 * @return CRC32 code.
 */
uint32_t crc32_bytewise(const uint8_t *buffer, size_t length, 
    const uint32_t *lu);

/**
 * @brief CRC32 lookup table initializer with polynomial generator CRC32 
 * (0x04C11DB7) and initial value 0x00000000.
 * @param lu Buffer of lookup table to initialize.
 * @return Lookup table pointer.
 */
uint32_t *generate_crc32_lu(uint32_t *lu);

/**
 * @brief Generate the string of the lookup table to hardcode from lookup 
 * table already initialized.
 * @param lu Lookup table already initialized.
 */
void print_crc32_lu(uint32_t *lu);

#ifdef __cplusplus
}
#endif

#endif // CRC_CRC_BYTEWISE_CRC32_BYTEWISE_H_
