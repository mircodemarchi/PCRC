/**
 * @author Mirco De Marchi
 * @date 2/08/2020
 * @brief Source file of binary modulo 2 arithmetic multiply operation.
 */

#include "mul2.h"
//------------------------------------------------------------------------------

uint16_t mul2_8(const uint8_t op1, const uint8_t op2)
{
    uint16_t ret = 0;
    for(uint8_t i = 0; i < (sizeof(uint8_t) * 8); i++)
    {
        if (op1 & (1U << i))
        {
            ret ^= (uint16_t) op2 << i;
        }
    }

    return ret;
}

uint32_t mul2_16(const uint16_t op1, const uint16_t op2)
{
    uint32_t ret = 0;
    for(uint8_t i = 0; i < (sizeof(uint16_t) * 8); i++)
    {
        if (op1 & (1U << i))
        {
            ret ^= (uint32_t) op2 << i;
        }
    }

    return ret;
}

uint64_t mul2_32(const uint32_t op1, const uint32_t op2)
{
    uint64_t ret = 0;
    for(uint8_t i = 0; i < (sizeof(uint32_t) * 8); i++)
    {
        if (op1 & (1U << i))
        {
            ret ^= (uint64_t) op2 << i;
        }
    }

    return ret;
}
