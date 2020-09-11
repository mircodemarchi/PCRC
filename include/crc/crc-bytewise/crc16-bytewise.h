/**
 * @author Mirco De Marchi
 * @date 11/09/2020
 * @brief Header file of CRC16 bytewise implementation.
 */

#ifndef CRC_CRC_BYTEWISE_CRC16_BYTEWISE_H_
#define CRC_CRC_BYTEWISE_CRC16_BYTEWISE_H_

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "generator.h"
//------------------------------------------------------------------------------

#ifdef __cplusplus
extern "C" {
#endif

extern const uint16_t crc16_lu[];
//------------------------------------------------------------------------------

/**
 * @brief CRC16 bytewise generic algorithm with lookup table already 
 * initialized.
 * @param buffer Message from which generate the CRC16 code.
 * @param length Size of the input message.
 * @param lu     Lookup table already initialized.
 * @return CRC16 code.
 */
uint16_t crc16_bytewise(const uint8_t *buffer, size_t length, 
    const uint16_t *lu);

/**
 * @brief CRC16 lookup table initializer with polynomial generator CRC16_CCITT 
 * (0x1012) and initial value 0x0000.
 * @param lu Buffer of lookup table to initialize.
 * @return Lookup table pointer.
 */
uint16_t *generate_crc16_lu(uint16_t *lu);

/**
 * @brief Generate the string of the lookup table to hardcode from lookup 
 * table already initialized.
 * @param lu Lookup table already initialized.
 */
void print_crc16_lu(uint16_t *lu);

#ifdef __cplusplus
}
#endif

#endif // CRC_CRC_BYTEWISE_CRC16_BYTEWISE_H_