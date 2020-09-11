/**
 * @author Mirco De Marchi
 * @date 31/07/2020
 * @brief Source file of CRC64 bitwise implementation.
 */

#include "crc64-bitwise.h"
//------------------------------------------------------------------------------

uint64_t crc64_bitwise(const uint8_t *buffer, size_t length)
{
    if (!buffer || length == 0) return 0xFFFFFFFF;

    const uint64_t generator = CRC64_ECMA; 
    uint64_t crc64 = 0; // Start CRC = 0x0000000000000000.

    for (size_t buffer_idx = 0; buffer_idx < length; buffer_idx++)
    {
        crc64 ^= ((uint64_t) *(buffer + buffer_idx)) << 56;

        for (int i = 0; i < 8; i++)
        {
            if ((crc64 & 0x8000000000000000) != 0)
            {
                crc64 = (uint64_t)((crc64 << 1) ^ generator);
            }
            else
            {
                crc64 <<= 1;
            }
        }
    }

    return crc64;
}
