/**
 * @author Mirco De Marchi
 * @date 23/07/2020
 * @brief Source file of binary modulo 2 arithmetic remainder operation.
 */

#include "mod2.h"
//------------------------------------------------------------------------------

uint8_t mod2_8(const uint8_t *divident, size_t length, uint8_t divisor)
{
    if (length == 0 || divident == NULL || divisor == 0) return 0xFF;
    
    uint8_t ret = 0;
    size_t divident_length = length * 8;

    // Find the first MSB bit of value 1 in divisor.
    uint8_t check_bit = 0x80;
    while ((divisor & check_bit) == 0) { check_bit >>= 1; }

    // Compute division for each byte.
    size_t buffer_idx = 0;
    size_t divisor_bit_idx = 0;
    while (divisor_bit_idx < divident_length)
    {
        for (uint8_t i = 0; i < 8; i++)
        {
            // Compute subtraction.
            if ((ret & check_bit) != 0)
            {
                ret = (uint8_t)(ret ^ divisor);
            }

            // Shift by 1 all the divident buffer.
            ret = (uint8_t)(ret << 1) 
                | (0x01 & (*(divident + buffer_idx) >> (8 - i - 1)));
            divisor_bit_idx++;
        }

        buffer_idx++;
    }

    // Compute the last subtraction.
    if ((ret & check_bit) != 0)
    {
        ret = (uint8_t)(ret ^ divisor);
    }

    return ret;
}

uint16_t mod2_16(const uint8_t *divident, size_t length, uint16_t divisor)
{
    if (length == 0 || divident == NULL || divisor == 0) return 0xFFFF;

    uint16_t ret = 0;
    size_t divident_length = length * 8;

    // Find the first MSB bit of value 1 in divisor.
    uint16_t check_bit = 0x8000;
    while ((divisor & check_bit) == 0) { check_bit >>= 1; }

    // Compute division for each byte.
    size_t buffer_idx = 0;
    size_t divisor_bit_idx = 0;
    while (divisor_bit_idx < divident_length)
    {
        for (uint8_t i = 0; i < 8; i++)
        {
            // Compute subtraction.
            if ((ret & check_bit) != 0)
            {
                ret = (uint16_t)(ret ^ divisor);
            }

            // Shift by 1 all the divident buffer.
            ret = (uint16_t)(ret << 1) 
                | (0x01 & (*(divident + buffer_idx) >> (8 - i - 1)));
            divisor_bit_idx++;
        }

        buffer_idx++;
    }

    // Compute the last subtraction.
    if ((ret & check_bit) != 0)
    {
        ret = (uint16_t)(ret ^ divisor);
    }

    return ret;
}

uint16_t mod2_16_u16(uint16_t divident, uint16_t divisor)
{
    if (divisor == 0) return 0xFFFF;

    uint16_t ret = 0;

    // Find the first MSB bit of value 1 in divisor.
    uint16_t check_bit = 0x8000;
    while ((divisor & check_bit) == 0) { check_bit >>= 1; }

    // Compute division for each byte.
    uint8_t divident_bit_length = (sizeof(uint16_t) * 8);
    for (uint8_t i = 0; i < divident_bit_length; i++)
    {
        // Compute subtraction.
        if ((ret & check_bit) != 0)
        {
            ret = (uint16_t)(ret ^ divisor);
        }

        // Shift by 1 all the divident buffer.
        ret = (uint16_t)(ret << 1) 
            | (0x0001 & (divident >> (divident_bit_length - i - 1)));
    }

    // Compute the last subtraction.
    if ((ret & check_bit) != 0)
    {
        ret = (uint16_t)(ret ^ divisor);
    }

    return ret;
}

uint32_t mod2_32(const uint8_t *divident, size_t length, uint32_t divisor)
{
    if (length == 0 || divident == NULL || divisor == 0) return 0xFFFFFFFF;

    uint32_t ret = 0;
    size_t divident_length = length * 8;

    // Find the first MSB bit of value 1 in divisor.
    uint32_t check_bit = 0x80000000;
    while ((divisor & check_bit) == 0) { check_bit >>= 1; }

    // Compute division for each byte.
    size_t buffer_idx = 0;
    size_t divisor_bit_idx = 0;
    while (divisor_bit_idx < divident_length)
    {
        for (uint8_t i = 0; i < 8; i++)
        {
            // Compute subtraction.
            if ((ret & check_bit) != 0)
            {
                ret = (uint32_t)(ret ^ divisor);
            }

            // Shift by 1 all the divident buffer.
            ret = (uint32_t)(ret << 1) 
                | (0x01 & (*(divident + buffer_idx) >> (8 - i - 1)));
            divisor_bit_idx++;
        }

        buffer_idx++;
    }

    // Compute the last subtraction.
    if ((ret & check_bit) != 0)
    {
        ret = (uint32_t)(ret ^ divisor);
    }

    return ret;
}

uint64_t mod2_64(const uint8_t *divident, size_t length, uint64_t divisor)
{
    if (length == 0 || divident == NULL || divisor == 0) 
        return 0xFFFFFFFFFFFFFFFF;
    
    uint64_t ret = 0;
    size_t divident_length = length * 8;

    // Find the first MSB bit of value 1 in divisor.
    uint64_t check_bit = 0x8000000000000000;
    while ((divisor & check_bit) == 0) { check_bit >>= 1; }

    // Compute division for each byte.
    size_t buffer_idx = 0;
    size_t divisor_bit_idx = 0;
    while (divisor_bit_idx < divident_length)
    {
        for (uint8_t i = 0; i < 8; i++)
        {
            // Compute subtraction.
            if ((ret & check_bit) != 0)
            {
                ret = (uint64_t)(ret ^ divisor);
            }

            // Shift by 1 all the divident buffer.
            ret = (uint64_t)(ret << 1) 
                | (0x01 & (*(divident + buffer_idx) >> (8 - i - 1)));
            divisor_bit_idx++;
        }

        buffer_idx++;
    }

    // Compute the last subtraction.
    if ((ret & check_bit) != 0)
    {
        ret = (uint64_t)(ret ^ divisor);
    }

    return ret;
}
