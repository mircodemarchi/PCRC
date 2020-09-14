/**
 * @file log.h
 * @author Mirco De Marchi
 * @date 24/07/2020
 * @brief Log definitions.
 * 
 * Set project variable DEBUG=1 to enable log and set DEBUG_LOG_LEVEL a value 
 * from 0 to 4 where each log level adds information from the previous level 
 * which are as follows:
 * - 0: log with no level specified;
 * - 1: error log;
 * - 2: warning log;
 * - 3: information log;
 * - 4: debug information log;
 * 
 * The default DEBUG_LOG_LEVEL is 3.
 * 
 * To disable logs set DEBUG=0.
 */

#ifndef LOG_H_
#define LOG_H_

#ifdef __cplusplus
#include <cstdio>
#else
#include <stdio.h>
#endif
#include <string.h>
#include <time.h>
//------------------------------------------------------------------------------

#ifdef __cplusplus
extern "C" {
#endif

/// @brief Enumeration that represents the log levels.
enum {
	LOG_LVL_NONE, 	  	// 0
	LOG_LVL_ERROR, 		// 1
	LOG_LVL_WARNING,  	// 2
	LOG_LVL_INFO,      	// 3
	LOG_LVL_DEBUG, 		// 4
	LOG_LVL_MAX 		// 5
};

/// @brief The default debug log level is the info log level (3).
#ifndef DEBUG_LOG_LEVEL
#define DEBUG_LOG_LEVEL LOG_LVL_INFO
#endif

///	Buffer length used to statically save information for logs.
/// @{
#define TIME_BUFFER_LENGTH 26
#define FILE_BUFFER_LENGTH 40
#define FUNC_BUFFER_LENGTH 40
/// @}
//------------------------------------------------------------------------------

#if DEBUG
/// @brief Static strings of log levels.
static const char log_level[][6] = {
	"     ",
	"ERROR",
	"WARN ",
	"INFO ",
	"DEBUG"
};

///	Buffer used to statically save information for logs.
/// @{
static char time_buffer[TIME_BUFFER_LENGTH] = {};
static char file_buffer[FILE_BUFFER_LENGTH] = {};
static char func_buffer[FUNC_BUFFER_LENGTH] = {};
/// @}
#endif
//------------------------------------------------------------------------------

/// @brief Get the basename from __FILE__.
#if defined(_WIN32) || defined(_WIN64)
 	#define __FILENAME__ (strrchr(__FILE__, '\\') ?								\
	 	strrchr(__FILE__, '\\') + 1 : __FILE__)
#else
 	#define __FILENAME__ (strrchr(__FILE__, '/') ? 								\
		strrchr(__FILE__, '/') + 1 : __FILE__)
#endif /* defined(_WIN32) || defined(_WIN64) */

/// @brief Return if a log should be printed at a specific log level.
#define LOG_SHOULD_DISPLAY( level ) 											\
	(DEBUG && (level) <= DEBUG_LOG_LEVEL && (level) < LOG_LVL_MAX)

#if DEBUG
/**
 * @brief Support function to return the current time with 
 * the format %d-%m-%Y %H:%M:%S.
 * @return Pointer to the current time string.
 */
static inline char *get_current_time() 
{
	struct tm* tm_info;
	time_t t = time(NULL);
	tm_info = localtime(&t);
	strftime(time_buffer, 26, "%d-%m-%Y %H:%M:%S", tm_info);
    return time_buffer;
}

/// @brief Log helper to print all log info and filter the right log level.
#define LOG_HELPER(level, fmt, ...) do { 										\
	if (LOG_SHOULD_DISPLAY(level)) {											\
		sprintf(file_buffer, "%-15s:%-4d", __FILENAME__, __LINE__);				\
		sprintf(func_buffer, "%s()", __func__);									\
		printf("%s %s [%-19s %-20s] " fmt,           			 				\
				get_current_time(), log_level[level], 							\
				file_buffer, func_buffer, ##__VA_ARGS__);						\
	} 																			\
} while(0)

/// Log functions divided for each log level.
/// @{
#define LOG(fmt, ...)  LOG_HELPER(LOG_LVL_NONE   , fmt, ##__VA_ARGS__)
#define LOGE(fmt, ...) LOG_HELPER(LOG_LVL_ERROR  , fmt, ##__VA_ARGS__)
#define LOGW(fmt, ...) LOG_HELPER(LOG_LVL_WARNING, fmt, ##__VA_ARGS__)
#define LOGI(fmt, ...) LOG_HELPER(LOG_LVL_INFO   , fmt, ##__VA_ARGS__)
#define LOGD(fmt, ...) LOG_HELPER(LOG_LVL_DEBUG  , fmt, ##__VA_ARGS__)
/// @}
#else 	// DEBUG
#define LOG(fmt, ...)
#define LOGE(fmt, ...)
#define LOGW(fmt, ...)
#define LOGI(fmt, ...)
#define LOGD(fmt, ...)
#endif 	// DEBUG

#ifdef __cplusplus
}
#endif

#endif 	// LOG_H_
