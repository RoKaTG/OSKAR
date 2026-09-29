#ifndef OSKAR_GPU_H_
#define OSKAR_GPU_H_

/**
 * @file oskar_gpu.h
 *
 * @brief Single entry point for the vendor GPU runtime.
 *
 * OSKAR targets CUDA and HIP from one source tree. The two runtimes are
 * one-to-one for everything OSKAR uses, so rather than duplicating every
 * call site this header pulls in whichever runtime is configured and, for
 * HIP, maps the CUDA spellings onto it.
 *
 * The CUDA spelling is the canonical one purely because it came first;
 * nothing here implies a preference for that platform. Call sites should
 * include this header instead of <cuda_runtime_api.h> and guard on
 * OSKAR_HAVE_GPU rather than on either vendor macro, so that adding a third
 * backend later touches this file and not the rest of the tree.
 *
 * Only symbols OSKAR actually uses are mapped. Adding a new runtime call
 * means adding its mapping here, which is deliberate: it keeps the
 * dependency surface visible and small.
 */

#include <oskar_global.h>

#if defined(OSKAR_HAVE_HIP)

#if defined(__HIPCC__) || defined(__HIP__)
/* Device pass, or a .cu handed to the HIP front end: the full runtime, which
 * needs clang to parse. */
#include <hip/hip_runtime.h>
#else
/* Ordinary host translation units, compiled by the system C/C++ compiler.
 * hip_runtime.h would drag in device-side headers that GCC cannot parse, so
 * take the host API surface only -- which is all these files call. */
#include <hip/hip_runtime_api.h>
#endif

/* Error handling and status. */
#define cudaError_t                        hipError_t
#define cudaSuccess                        hipSuccess
#define cudaErrorInvalidConfiguration      hipErrorInvalidConfiguration
#define cudaGetErrorString                 hipGetErrorString
#define cudaPeekAtLastError                hipPeekAtLastError

/* Device management. */
#define cudaGetDevice                      hipGetDevice
#define cudaSetDevice                      hipSetDevice
#define cudaGetDeviceCount                 hipGetDeviceCount
#define cudaDeviceReset                    hipDeviceReset
#define cudaDeviceSynchronize              hipDeviceSynchronize
#define cudaGetDeviceProperties            hipGetDeviceProperties
#define cudaDeviceGetAttribute             hipDeviceGetAttribute
#define cudaDriverGetVersion               hipDriverGetVersion
#define cudaRuntimeGetVersion              hipRuntimeGetVersion

/* Device attributes. */
#define cudaDevAttrClockRate               hipDeviceAttributeClockRate
#define cudaDevAttrMemoryClockRate         hipDeviceAttributeMemoryClockRate
#define cudaDevAttrComputeCapabilityMajor  hipDeviceAttributeComputeCapabilityMajor
#define cudaDevAttrComputeCapabilityMinor  hipDeviceAttributeComputeCapabilityMinor

/* Memory. */
#define cudaMalloc                         hipMalloc
#define cudaFree                           hipFree
#define cudaMemcpy                         hipMemcpy
#define cudaMemset                         hipMemset
#define cudaMemGetInfo                     hipMemGetInfo
#define cudaMemcpyHostToDevice             hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost             hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice           hipMemcpyDeviceToDevice

/* Events. */
#define cudaEvent_t                        hipEvent_t
#define cudaEventCreate                    hipEventCreate
#define cudaEventDestroy                   hipEventDestroy
#define cudaEventRecord                    hipEventRecord
#define cudaEventSynchronize               hipEventSynchronize
#define cudaEventElapsedTime               hipEventElapsedTime

/* Kernel launch. */
#define cudaLaunchKernel                   hipLaunchKernel

/* Device properties. CUDA spells this `struct cudaDeviceProp`, HIP a typedef
 * `hipDeviceProp_t` of a struct with a different tag, so a plain macro swap
 * would leave `struct hipDeviceProp_t` -- which does not name a type. Call
 * sites use oskar_GpuDeviceProp instead. */
typedef hipDeviceProp_t oskar_GpuDeviceProp;

#elif defined(OSKAR_HAVE_CUDA)

#include <cuda_runtime_api.h>

typedef struct cudaDeviceProp oskar_GpuDeviceProp;

#endif /* backend selection */

#endif /* include guard */
