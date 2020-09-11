/**
 * @author Mirco De Marchi
 * @date 23/07/2020
 * @brief Source of test CRC utils.
 */

#include "test-utils.h"

#include "log.h"
//------------------------------------------------------------------------------

void log_processed_data_8(const uint8_t *buffer, size_t length) 
{
    LOGD("Processed data: [");
    if (LOG_SHOULD_DISPLAY(LOG_LVL_DEBUG)) {
        size_t i;
        for (i = 0; i < length - 1; i++) 
        {
            printf("0x%x ", (uint8_t) *(buffer + i));
        }
        printf("0x%x]\n", (uint8_t) *(buffer + i));
    }
}

void log_processed_data_16(const uint16_t *buffer, size_t length) 
{
    LOGD("Processed data: [");
    if (LOG_SHOULD_DISPLAY(LOG_LVL_DEBUG)) {
        size_t i;
        for (i = 0; i < length - 1; i++) 
        {
            printf("0x%x ", (uint16_t) *(buffer + i));
        }
        printf("0x%x]\n", (uint16_t) *(buffer + i));
    }
}

void log_processed_data_32(const uint32_t *buffer, size_t length) 
{
    LOGD("Processed data: [");
    if (LOG_SHOULD_DISPLAY(LOG_LVL_DEBUG)) {
        size_t i;
        for (i = 0; i < length - 1; i++) 
        {
            printf("0x%x ", (uint32_t) *(buffer + i));
        }
        printf("0x%x]\n", (uint32_t) *(buffer + i));
    }
}

void log_processed_data_64(const uint64_t *buffer, size_t length) 
{
    LOGD("Processed data: [");
    if (LOG_SHOULD_DISPLAY(LOG_LVL_DEBUG)) {
        size_t i;
        for (i = 0; i < length - 1; i++) 
        {
            printf("0x%lx ", (uint64_t) *(buffer + i));
        }
        printf("0x%lx]\n", (uint64_t) *(buffer + i));
    }
}