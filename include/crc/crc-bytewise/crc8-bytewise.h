/**
 * @author Mirco De Marchi
 * @date 11/09/2020
 * @brief Header file of CRC8 bytewise implementation.
 */

#ifndef CRC_CRC_BYTEWISE_CRC8_BYTEWISE_H_
#define CRC_CRC_BYTEWISE_CRC8_BYTEWISE_H_

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "generator.h"
//------------------------------------------------------------------------------

#ifdef __cplusplus
extern "C" {
#endif

extern const uint8_t crc8_lu[];
//------------------------------------------------------------------------------

/**
 * @brief CRC8 bytewise generic algorithm with lookup table already 
 * initialized.
 * @param buffer Message from which generate the CRC8 code.
 * @param length Size of the input message.
 * @param lu     Lookup table already initialized.
 * @return CRC8 code.
 */
uint8_t crc8_bytewise(const uint8_t *buffer, size_t length, 
    const uint8_t *lu);

/**
 * @brief CRC8 lookup table initializer with polynomial generato value CRC8_SAE 
 * (0x1D) and initial value 0x00.
 * @param lu Buffer of lookup table to initialize.
 * @return Lookup table pointer.
 */
uint8_t *generate_crc8_lu(uint8_t *lu);

/**
 * @brief Generate the string of the lookup table to hardcode from lookup 
 * table already initialized.
 * @param lu Lookup table already initialized.
 */
void print_crc8_lu(uint8_t *lu);

#ifdef __cplusplus
}
#endif

#endif // CRC_CRC_BYTEWISE_CRC8_BYTEWISE_H_
