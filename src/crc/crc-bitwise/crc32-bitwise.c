/**
 * @author Mirco De Marchi
 * @date 23/07/2020
 * @brief Source file of CRC32 bitwise implementation.
 */

#include "crc32-bitwise.h"

#include <string.h>

#include "mod2.h"
//------------------------------------------------------------------------------

uint32_t crc32_bitwise(const uint8_t *buffer, size_t length)
{
    if (!buffer || length == 0) return 0xFFFFFFFF;

#if CRC_USE_MOD2
    uint8_t *running = (uint8_t *) malloc(length + sizeof(uint32_t));
    memset(running, 0x00, length + sizeof(uint32_t));
    memcpy(running, buffer, length);

    uint32_t ret = (uint32_t) mod2_64(running, length + sizeof(uint32_t), 
                                      CRC32 + 0x100000000);
    free(running);
    return ret;
#else
    const uint32_t generator = CRC32; 
    uint32_t crc32 = 0; // Start CRC = 0x00.

    for (size_t buffer_idx = 0; buffer_idx < length; buffer_idx++)
    {
        crc32 ^= (uint32_t)(*(buffer + buffer_idx) << 24);

        for (int i = 0; i < 8; i++)
        {
            if ((crc32 & 0x80000000) != 0)
            {
                crc32 = (uint32_t)((crc32 << 1) ^ generator);
            }
            else
            {
                crc32 <<= 1;
            }
        }
    }

    return crc32;
#endif
}
