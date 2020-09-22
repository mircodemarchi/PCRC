cmake_minimum_required(VERSION 3.8)
message(STATUS "CMake version: ${CMAKE_VERSION}")

# ------------------------------------------------------------------------------
# Setup project.

# Project with not optional CUDA.
project(PCRC LANGUAGES CUDA CXX C)
set(PROJECT_NAME PCRC)

# Check if CUDA is available.
include(CheckLanguage)
check_language(CUDA)

# Check if a compiler exists.
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "Cuda compiler has been found: ${CMAKE_CUDA_COMPILER}")
else(CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "Cuda compiler has not been found")
endif()

# Check host compiler.
if(DEFINED ${CMAKE_CUDA_HOST_COMPILER})
    message(STATUS "Cuda host compiler defined: ${CMAKE_CUDA_HOST_COMPILER}")
endif()

# Check correct compiler id.
if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    message(STATUS "Cuda compiler id match NVIDIA")
else(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    message(FATAL_ERROR "Cuda compiler id doesn't match")
endif()

message(STATUS 
        "CMake setup project succeeds with NVIDIA "
        "v${CMAKE_CUDA_COMPILER_VERSION}")

# Debug mode and logs.
set(DEBUG true CACHE BOOL "Enable debug log")
set(DEBUG_LOG_LEVEL 5 CACHE INTEGER "Logging level")
if(NOT DEBUG_LOG_LEVEL MATCHES "^[0-9]+$")
    message(FATAL_ERROR "DEBUG_LOG_LEVEL must be a number")
endif()

if(DEBUG)
    add_definitions(-DDEBUG=1 -DDEBUG_LOG_LEVEL=${DEBUG_LOG_LEVEL})
    message(STATUS "Enable DEBUG mode with DEBUG_LOG_LEVEL=${DEBUG_LOG_LEVEL}")
else(DEBUG)
    add_definitions(-DDEBUG=0 -DDEBUG_LOG_LEVEL=0)
    message(STATUS "Disable DEBUG mode")
endif(DEBUG)

# Compile test directory.
option(TEST "Enable test compile" ON)

# CRC use binary module 2 for calulation option.
set(CRC_USE_MOD2 false CACHE BOOL "Use binary module 2 to calculate crc")
if(CRC_USE_MOD2)
    message(STATUS "Enable the use of binary module 2 to calculate crc")
else(CRC_USE_MOD2)
    message(STATUS "Disable the use of binary module 2 to calculate crc")
endif(CRC_USE_MOD2)

# ------------------------------------------------------------------------------
# Set the C++ compiler options.
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O1")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fomit-frame-pointer")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-builtin")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-stack-protector")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=i686")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m32")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wpedantic")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic-errors")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wconversion")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wcast-align")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wcast-qual")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wdisabled-optimization")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wfloat-equal")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wfloat-conversion")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wformat=2")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wformat-nonliteral")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wformat-security")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wformat-y2k")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wimport")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Winit-self")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Winline")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Winvalid-pch")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-long-long")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wmissing-field-initializers")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wmissing-format-attribute")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wmissing-include-dirs")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wpacked")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wpointer-arith")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wredundant-decls")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wshadow")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wstack-protector")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wstrict-aliasing=2")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wswitch")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wswitch-default")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wswitch-enum")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wunreachable-code")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wunused")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wunused-function")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wunused-label")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wunused-parameter")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wunused-value")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wunused-variable")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wvariadic-macros")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wwrite-strings")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wsign-compare")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wsign-conversion")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wuninitialized")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fmessage-length=0")

# ------------------------------------------------------------------------------
# Set the C compiler options.
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -nostdlib")
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -nostdinc")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g3")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ggdb")
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fomit-frame-pointer")
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-builtin")
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-stack-protector")
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=i686")
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -m32")
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu99")
set(CMAKE_C_FLAGS"${CMAKE_C_FLAGS} -Wall")
#set(CMAKE_C_FLAGS"${CMAKE_C_FLAGS} -Wpedantic")
#set(CMAKE_C_FLAGS"${CMAKE_C_FLAGS} -pedantic-errors")
#set(CMAKE_C_FLAGS"${CMAKE_C_FLAGS} -Wextra")
#set(CMAKE_C_FLAGS"${CMAKE_C_FLAGS} -Werror")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wconversion")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wcast-align")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wcast-qual")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wdisabled-optimization")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wfloat-equal")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wfloat-conversion")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wformat=2")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wformat-nonliteral")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wformat-security")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wformat-y2k")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wimport")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Winit-self")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Winline")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Winvalid-pch")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-long-long")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wmissing-field-initializers")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wmissing-format-attribute")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wmissing-include-dirs")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wpacked")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wpointer-arith")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wredundant-decls")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wshadow")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wstack-protector")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wstrict-aliasing=2")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wswitch")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wswitch-default")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wswitch-enum")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wunreachable-code")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wunused")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wunused-function")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wunused-label")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wunused-parameter")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wunused-value")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wunused-variable")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wvariadic-macros")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wwrite-strings")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wsign-compare")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wsign-conversion")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wuninitialized")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fmessage-length=0")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -msse4")

# ------------------------------------------------------------------------------
# Set the CUDA compiler options.
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_30")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_70")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_50,code=sm_50")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_52,code=sm_52")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_60,code=sm_60")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_61,code=sm_61")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_70,code=sm_70")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_75,code=sm_75")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_75,code=compute_75")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -w")

# ------------------------------------------------------------------------------
# Add the includes.
include_directories(
    include
    include/parco
    include/crc
    include/util
    include/crc/crc-bitwise
    include/crc/crc-bytewise
    include/device/examples/
    include/device/pcrc/
    include/mod2-arithmetic/
)

if(TEST)
    include_directories(
        test/include
    )
endif(TEST)

# ------------------------------------------------------------------------------
# Add the source files.
set(COMMON_SOURCE_FILES
    src/crc/crc-bitwise/crc8-bitwise.c
    src/crc/crc-bitwise/crc16-bitwise.c
    src/crc/crc-bitwise/crc32-bitwise.c
    src/crc/crc-bitwise/crc64-bitwise.c

    src/crc/crc-bytewise/crc8-bytewise.c
    src/crc/crc-bytewise/crc16-bytewise.c
    src/crc/crc-bytewise/crc32-bytewise.c
    src/crc/crc-bytewise/crc64-bytewise.c
    
    src/mod2-arithmetic/mod2.c
    src/mod2-arithmetic/mul2.c

    src/crc/crc-utils.c
)

if(TEST)
    set(TEST_SOURCE_FILES  
        test/src/test.c
        test/src/test-crc-bitwise.c
        test/src/test-crc-bytewise.c
        test/src/test-mod2-arithmetic.c
        test/src/test-utils.c
    )
endif(TEST)

set(CUDA_SOURCE_FILES
    src/main.cu
    src/util/compare-run.cu
    src/device/examples/matrix-multiplication.cu
    src/device/pcrc/pcrc8.cu
    src/device/pcrc/pcrc16.cu
    src/device/pcrc/pcrc32.cu
)

# ------------------------------------------------------------------------------
# Add the executable.
set(MAIN_EXECUTABLE_NAME ${PROJECT_NAME}.x)
set(TEST_EXECUTABLE_NAME ${PROJECT_NAME}-test.x)

set(MAIN_SOURCE
    ${COMMON_SOURCE_FILES}
    ${CUDA_SOURCE_FILES}
)
add_executable(${MAIN_EXECUTABLE_NAME} 
    ${MAIN_SOURCE}
)

if(TEST)
    set(TEST_SOURCE
        ${COMMON_SOURCE_FILES}
        ${TEST_SOURCE_FILES}
    )
    add_executable(${TEST_EXECUTABLE_NAME} 
        ${TEST_SOURCE}
    )
endif(TEST)

# ------------------------------------------------------------------------------
# Add custom commands.
add_custom_target(
    run
    COMMAND ${MAIN_EXECUTABLE_NAME}
    DEPENDS ${MAIN_EXECUTABLE_NAME}
)

if(TEST)
    add_custom_target(
        run-test
        COMMAND ${TEST_EXECUTABLE_NAME}
        DEPENDS ${TEST_EXECUTABLE_NAME}
    )
endif(TEST)