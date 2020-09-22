/**
 * @author Mirco De Marchi
 * @date 23/07/2020
 * @brief CRC standard polynomial generator.
 */

#ifndef CRC_CRC_BITWISE_GENERATOR_H_
#define CRC_CRC_BITWISE_GENERATOR_H_

#ifdef __cplusplus
extern "C" {
#endif

/// CRC8 polynomial generator.
#define CRC8_SAE 0x1D

/// CRC16 polynomial generator.
#define CRC16_CCITT 0x1021

/// CRC32 polynomial generator.
#define CRC32 0x04C11DB7
#define CRC32_INTEL 0x1EDC6F41

/// CRC64 polynomial generator.
#define CRC64_ECMA 0x42F0E1EBA9EA3693

#ifdef __cplusplus
}
#endif

#endif  // CRC_CRC_BITWISE_GENERATOR_H_
