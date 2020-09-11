/**
 * @author Mirco De Marchi
 * @date 23/07/2020
 * @brief Source file of CRC16 bitwise implementation.
 */

#include "crc16-bitwise.h"

#include <string.h>

#include "mod2.h"
//------------------------------------------------------------------------------

uint16_t crc16_bitwise(const uint8_t *buffer, size_t length)
{
    if (!buffer || length == 0) return 0xFFFF;

#if CRC_USE_MOD2
    uint8_t *running = (uint8_t *) malloc(length + sizeof(uint16_t));
    memset(running, 0x00, length + sizeof(uint16_t));
    memcpy(running, buffer, length);

    uint16_t ret = (uint16_t) mod2_32(running, length + sizeof(uint16_t), 
                                      CRC16_CCITT + 0x10000);
    free(running);
    return ret;
#else
    const uint16_t generator = CRC16_CCITT;
    uint16_t crc16 = 0; // Start CRC = 0x00.

    for (size_t buffer_idx = 0; buffer_idx < length; buffer_idx++)
    {
        crc16 ^= (uint16_t)(*(buffer + buffer_idx) << 8);

        for (int i = 0; i < 8; i++)
        {
            if ((crc16 & 0x8000) != 0) 
            {
                crc16 = (uint16_t)((crc16 << 1) ^ generator);
            }
            else
            {
                crc16 <<= 1;
            }
        }
    }

    return crc16;
#endif
}
