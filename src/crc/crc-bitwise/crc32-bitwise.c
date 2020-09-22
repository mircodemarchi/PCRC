/**
 * @author Mirco De Marchi
 * @date 23/07/2020
 * @brief Source file of CRC32 bitwise implementation.
 */

#include "crc32-bitwise.h"

#include <string.h>
#include <nmmintrin.h>

#include "mod2.h"
#include "crc-utils.h"
//------------------------------------------------------------------------------

uint32_t crc32_std_bitwise(uint32_t crc, const uint8_t *buffer, 
                           size_t length, const uint32_t gen)
{
    if (!buffer || length == 0) return 0xFFFFFFFF;

#if CRC_USE_MOD2
    uint8_t *running = (uint8_t *) malloc(length + sizeof(uint32_t));
    memset(running, 0x00, length + sizeof(uint32_t));
    memcpy(running, buffer, length);

    uint32_t ret = (uint32_t) mod2_64(running, length + sizeof(uint32_t), 
                                      gen + 0x100000000);
    free(running);
    return ret;
#else

    for (size_t buffer_idx = 0; buffer_idx < length; buffer_idx++)
    {
        crc ^= (uint32_t)(*(buffer + buffer_idx) << 24);

        for (int i = 0; i < 8; i++)
        {
            if ((crc & 0x80000000) != 0)
            {
                crc = (uint32_t)((crc << 1) ^ gen);
            }
            else
            {
                crc <<= 1;
            }
        }
    }

    return crc;
#endif
}

uint32_t crc32_rev_bitwise(uint32_t crc, const uint8_t *buffer, 
                           size_t length, const uint32_t gen)
{
    if (!buffer || length == 0) return 0xFFFFFFFF;

    crc = reverse32(crc);

#if CRC_USE_MOD2
    uint8_t *running = (uint8_t *) malloc(length + sizeof(uint32_t));
    memset(running, 0x00, length + sizeof(uint32_t));

    // memcpy reversed
    memcpy_rev(running, buffer, length);

    uint32_t ret = (uint32_t) mod2_64(running, length + sizeof(uint32_t), 
                                      gen + 0x100000000);
    free(running);
    return reverse32(ret);
#else

    for (size_t buffer_idx = 0; buffer_idx < length; buffer_idx++)
    {
        crc ^= (uint32_t)(reverse8_lu[*(buffer + buffer_idx)] << 24);

        for (int i = 0; i < 8; i++)
        {
            if ((crc & 0x80000000) != 0)
            {
                crc = (uint32_t)((crc << 1) ^ gen);
            }
            else
            {
                crc <<= 1;
            }
        }
    }

    return reverse32(crc);
#endif
}

uint32_t crc32_rev_hw_bitwise(uint32_t crc, const uint8_t *buffer, 
                              size_t length, const uint32_t gen)
{
    if (!buffer || length == 0) return 0xFFFFFFFF;

    for (size_t buffer_idx = 0; buffer_idx < length; buffer_idx++)
    {
        crc = _mm_crc32_u8(crc, *(buffer + buffer_idx));
    }

    return crc;
}

uint32_t crc32_bitwise(const uint8_t *buffer, size_t length)
{
    return crc32_std_bitwise(0x00000000, buffer, length, CRC32);
}

uint32_t crc32_intel_bitwise(uint32_t crc, const uint8_t *buffer, size_t length)
{
    return crc32_rev_bitwise(crc, buffer, length, CRC32_INTEL);
}

uint32_t crc32_intel_hw_bitwise(uint32_t crc, const uint8_t *buffer, size_t length)
{
    return crc32_rev_hw_bitwise(crc, buffer, length, CRC32_INTEL);
}