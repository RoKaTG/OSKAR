/*
 * Copyright (c) 2011-2019, The University of Oxford.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the University of Oxford nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef OSKAR_VECTOR_TYPES_H_
#define OSKAR_VECTOR_TYPES_H_

/**
 * @file oskar_vector_types.h
 */

#ifdef __CUDACC__
/* Include the CUDA vector types header first, if we're compiling with nvcc. */
#   include <vector_types.h>
#elif defined(__HIPCC__) || defined(__HIP__)
/* HIP device code: HIP defines float2 and double2 as HIP_vector_type, along
 * with __align__. Take theirs, exactly as we take CUDA's above. */
#   include <hip/hip_runtime.h>
#elif defined(OSKAR_HAVE_HIP) && defined(__HIP_PLATFORM_AMD__)
/* Host code inside the OSKAR library in a HIP build. Any HIP host header --
 * hip_runtime_api.h via channel_descriptor.h, hipfft.h via hip_complex.h --
 * declares float2 and double2 too, in both C and C++, and a second
 * declaration of the same name is a hard error. Pulling in the runtime API
 * here, ahead of our own definitions, makes HIP's the ones in force in every
 * such translation unit regardless of include order. __HIP_PLATFORM_AMD__ is
 * supplied by the hip::host target, so this only applies to sources that
 * actually link against HIP.
 *
 * In C, HIP's types are plain structs with natural alignment (4 and 8 bytes)
 * rather than the 8 and 16 used below and by CUDA. The sizes are identical,
 * so array strides and the member offsets of float4c/double4c are unchanged,
 * and on the x86-64 SysV ABI these structs are passed and returned in the
 * same SSE registers either way. In C++ HIP's layout matches exactly. */
#   include <hip/hip_runtime_api.h>
#endif

/* Memory alignment macros mirroring those used by CUDA. Defined only where
 * the vendor headers have not already supplied them: HIP provides __align__
 * from its full device runtime but not from the host API headers. */
#ifndef __align__
#   if defined(__GNUC__)
#       define __align__(n) __attribute__((aligned(n)))
#   elif defined(_MSC_VER)
#       define __align__(n) __declspec(align(n))
#   endif
#endif
#ifndef __builtin_align__
#   if defined(__GNUC__) || defined(_WIN64)
#       define __builtin_align__(a) __align__(a)
#   else
#       define __builtin_align__(a)
#   endif
#endif

/* Our own float2/double2, unless a vendor's are already in scope. The last
 * test is HIP's include guard: it is what catches a host translation unit
 * that reached HIP's vector types through some route other than the include
 * above. */
#if !(defined(__VECTOR_TYPES_H__) || defined(__CUDACC__) || \
        defined(__HIPCC__) || defined(__HIP__) || \
        defined(HIP_INCLUDE_HIP_AMD_DETAIL_HIP_VECTOR_TYPES_H))

/**
 * @brief Two-element structure (single precision).
 *
 * @details
 * Structure used to hold data for a length-2 vector.
 * This must be compatible with the CUDA float2 type.
 */
struct __builtin_align__(8) float2 { float x, y; };
typedef struct float2 float2;

/**
 * @brief Two-element structure (double precision).
 *
 * @details
 * Structure used to hold data for a length-2 vector.
 * This must be compatible with the CUDA double2 type.
 */
struct __builtin_align__(16) double2 { double x, y; };
typedef struct double2 double2;
#endif

/**
 * @brief Four-element complex structure (single precision).
 *
 * @details
 * Structure used to hold data for a length-4 single precision complex vector.
 * When used as a matrix, the elements should be interpreted as:
 *
 *   ( a  b )
 *   ( c  d )
 */
struct __align__(32) float4c { float2 a, b, c, d; };
typedef struct float4c float4c;

/**
 * @brief Four-element complex structure (double precision).
 *
 * @details
 * Structure used to hold data for a length-4 double precision complex vector.
 * When used as a matrix, the elements should be interpreted as:
 *
 *   ( a  b )
 *   ( c  d )
 */
struct __align__(64) double4c { double2 a, b, c, d; };
typedef struct double4c double4c;

#endif /* include guard */
