/**
 * @author Mirco De Marchi
 * @date 23/07/2020
 * @brief Source file of CRC8 bitwise implementation.
 */

#include "crc8-bitwise.h"

#include <string.h>

#include "mod2.h"
//------------------------------------------------------------------------------

uint8_t crc8_bitwise(const uint8_t *buffer, size_t length)
{
    if (!buffer || length == 0) return 0xFF;

#if CRC_USE_MOD2
    uint8_t *running = (uint8_t *) malloc(length + sizeof(uint8_t));
    memset(running, 0x00, length + sizeof(uint8_t));
    memcpy(running, buffer, length);

    uint8_t ret = (uint8_t) mod2_16(running, length + sizeof(uint8_t), 
                                    CRC8_SAE + 0x100);
    free(running);
    return ret;
#else
    const uint8_t generator = CRC8_SAE;
    uint8_t crc8 = 0; // Start CRC = 0x00.

    for (size_t buffer_idx = 0; buffer_idx < length; buffer_idx++)
    {
        crc8 ^= *(buffer + buffer_idx); 

        for (int i = 0; i < 8; i++)
        {
            if ((crc8 & 0x80) != 0)
            {
                crc8 = (uint8_t)((crc8 << 1) ^ generator);
            }
            else
            {
                crc8 <<= 1;
            }
        }
    }

    return crc8;
#endif
}
