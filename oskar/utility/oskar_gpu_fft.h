#ifndef OSKAR_GPU_FFT_H_
#define OSKAR_GPU_FFT_H_

/**
 * @file oskar_gpu_fft.h
 *
 * @brief Single entry point for the vendor GPU FFT library.
 *
 * hipFFT is a thin wrapper over rocFFT that deliberately mirrors the cuFFT
 * interface, so the mapping below is direct. See oskar_gpu.h for the
 * reasoning behind keeping the CUDA spelling as canonical.
 */

#include <oskar_global.h>

#if defined(OSKAR_HAVE_HIP)

#include <hipfft/hipfft.h>

/* Handles, types and results. */
#define cufftHandle             hipfftHandle
#define cufftResult             hipfftResult
#define cufftComplex            hipfftComplex
#define cufftDoubleComplex      hipfftDoubleComplex

/* Transform types and directions. */
#define CUFFT_C2C               HIPFFT_C2C
#define CUFFT_Z2Z               HIPFFT_Z2Z
#define CUFFT_FORWARD           HIPFFT_FORWARD

/* Plan and execution. */
#define cufftPlan1d             hipfftPlan1d
#define cufftPlan2d             hipfftPlan2d
#define cufftExecC2C            hipfftExecC2C
#define cufftExecZ2Z            hipfftExecZ2Z
#define cufftDestroy            hipfftDestroy

/* Result codes. */
#define CUFFT_SUCCESS           HIPFFT_SUCCESS
#define CUFFT_INVALID_PLAN      HIPFFT_INVALID_PLAN
#define CUFFT_ALLOC_FAILED      HIPFFT_ALLOC_FAILED
#define CUFFT_INTERNAL_ERROR    HIPFFT_INTERNAL_ERROR
#define CUFFT_EXEC_FAILED       HIPFFT_EXEC_FAILED
#define CUFFT_SETUP_FAILED      HIPFFT_SETUP_FAILED
#define CUFFT_INVALID_SIZE      HIPFFT_INVALID_SIZE
#define CUFFT_INVALID_VALUE     HIPFFT_INVALID_VALUE
#define CUFFT_UNALIGNED_DATA    HIPFFT_UNALIGNED_DATA
#define CUFFT_NO_WORKSPACE      HIPFFT_NO_WORKSPACE

#elif defined(OSKAR_HAVE_CUDA)

#include <cufft.h>

#endif /* backend selection */

#endif /* include guard */
