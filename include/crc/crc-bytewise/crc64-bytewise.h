/**
 * @author Mirco De Marchi
 * @date 11/09/2020
 * @brief Header file of CRC64 bytewise implementation.
 */

#ifndef CRC_CRC_BYTEWISE_CRC64_BYTEWISE_H_
#define CRC_CRC_BYTEWISE_CRC64_BYTEWISE_H_

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "generator.h"
//------------------------------------------------------------------------------

#ifdef __cplusplus
extern "C" {
#endif

extern const uint64_t crc64_lu[];
//------------------------------------------------------------------------------

/**
 * @brief CRC64 bytewise generic algorithm with lookup table already 
 * initialized.
 * @param buffer Message from which generate the CRC64 code.
 * @param length Size of the input message.
 * @param lu     Lookup table already initialized.
 * @return CRC64 code.
 */
uint64_t crc64_bytewise(const uint8_t *buffer, size_t length, 
    const uint64_t *lu);

/**
 * @brief CRC64 lookup table initializer with polynomial generator CRC64 
 * (0x42F0E1EBA9EA3693) and initial value 0x0000000000000000.
 * @param lu Buffer of lookup table to initialize.
 * @return Lookup table pointer.
 */
uint64_t *generate_crc64_lu(uint64_t *lu);

/**
 * @brief Generate the string of the lookup table to hardcode from lookup 
 * table already initialized.
 * @param lu Lookup table already initialized.
 */
void print_crc64_lu(uint64_t *lu);

#ifdef __cplusplus
}
#endif

#endif // CRC_CRC_BYTEWISE_CRC64_BYTEWISE_H_
