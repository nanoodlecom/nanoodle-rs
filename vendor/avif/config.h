/* dav1d's portable C paths. All architecture-specific DSP dispatch is disabled.
 * A toolchain with C11 atomics is required. MSVC uses dav1d's compatibility
 * header; see the pack README. */
#pragma once
#define HAVE_ASM 0
#define CONFIG_8BPC 1
#define CONFIG_16BPC 1
#define CONFIG_LOG 0
#define TRIM_DSP_FUNCTIONS 1
#define HAVE_C11_GENERIC 1
#define DAV1D_API
#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
#define ENDIANNESS_BIG 1
#else
#define ENDIANNESS_BIG 0
#endif
#if defined(_WIN32)
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0601
#endif
#ifndef UNICODE
#define UNICODE 1
#endif
#ifndef _UNICODE
#define _UNICODE 1
#endif
#define HAVE_IO_H 1
#else
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#define HAVE_POSIX_MEMALIGN 1
#define HAVE_SYS_TYPES_H 1
#define HAVE_UNISTD_H 1
#endif
