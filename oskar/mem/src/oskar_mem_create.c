/*
 * Copyright (c) 2013-2026, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <stdlib.h>
#include <string.h>

#ifdef OSKAR_HAVE_GPU
#include "utility/oskar_gpu.h"
#endif

#include "mem/oskar_mem.h"
#include "mem/private_mem.h"
#include "utility/oskar_device.h"

#define MAX(a, b) ((a) > (b) ? (a) : (b))

#ifdef __cplusplus
extern "C" {
#endif


oskar_Mem* oskar_mem_create(
        int type,
        int location,
        size_t num_elements,
        int* status
)
{
    oskar_Mem* mem = (oskar_Mem*) calloc(1, sizeof(oskar_Mem));
    if (!mem)
    {
        *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;         /* LCOV_EXCL_LINE */
        return 0;                                         /* LCOV_EXCL_LINE */
    }

    /* Initialise meta-data.
     * (This must happen regardless of the status code.) */
    mem->type = type;
    mem->location = location;
    mem->num_elements = 0;
    mem->owner = 1;
    mem->data = NULL;
    mem->ref_count = 1;
    mem->mutex = oskar_mutex_create();

    /* Check if allocation should happen or not. */
    if (!status || *status || num_elements == 0)
    {
        return mem;
    }

    /* Get the memory size. */
    const size_t element_size = oskar_mem_element_size(type);
    if (element_size == 0)
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return mem;
    }

    /* Check whether the memory should be on the host or the device. */
    mem->num_elements = num_elements;
    if (location == OSKAR_CPU)
    {
        /* Allocate host memory, suitably aligned to the element size. */
#ifdef OSKAR_OS_WIN
        const size_t bytes = num_elements * element_size;
        mem->data = _aligned_malloc(bytes, element_size);
        if (mem->data) memset(mem->data, 0, bytes);
#elif defined(OSKAR_OS_MAC) || _POSIX_C_SOURCE >= 200112L
        const size_t bytes = num_elements * element_size;
        const size_t alignment = MAX(element_size, 16);
        const int error = posix_memalign(&mem->data, alignment, bytes);
        if (error != 0)
        {
            *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;     /* LCOV_EXCL_LINE */
            return mem;                                   /* LCOV_EXCL_LINE */
        }
        else if (mem->data)
        {
            memset(mem->data, 0, bytes);
        }
#else
        mem->data = calloc(num_elements, element_size);
#endif
        if (mem->data == NULL)
        {
            *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;     /* LCOV_EXCL_LINE */
            return mem;                                   /* LCOV_EXCL_LINE */
        }
    }
    else if (location == OSKAR_GPU)
    {
#ifdef OSKAR_HAVE_GPU
        /* Allocate GPU memory. On CUDA, don't clear it, for efficiency. */
        const size_t bytes = num_elements * element_size;
        *status = (int) cudaMalloc(&mem->data, bytes);
        if (!*status && mem->data == NULL)
        {
            *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;     /* LCOV_EXCL_LINE */
        }
#ifdef OSKAR_HAVE_HIP
        /* Clear new device allocations on HIP. Without this, the first kernel
         * in a process can read incorrect values from a buffer that was
         * filled by a host-to-device copy, on ROCm 6.1 with gfx90a. CUDA does
         * not need it: fresh device memory reads as zero there in practice.
         * Set OSKAR_HIP_NO_CLEAR_ON_ALLOC=1 to disable, to check whether a
         * given ROCm version still requires it. */
        if (!*status && bytes > 0 && !getenv("OSKAR_HIP_NO_CLEAR_ON_ALLOC"))
        {
            *status = (int) cudaMemset(mem->data, 0, bytes);
        }
#endif
#else
        *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;           /* LCOV_EXCL_LINE */
#endif
    }
    else if (location & OSKAR_CL)
    {
#ifdef OSKAR_HAVE_OPENCL
        /* Allocate OpenCL memory buffer using the current context. */
        cl_int error = 0;
        const size_t bytes = num_elements * element_size;
        mem->buffer = clCreateBuffer(
                oskar_device_context_cl(),
                CL_MEM_READ_WRITE, bytes, NULL, &error
        );
        if (error != CL_SUCCESS)
        {
            *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;     /* LCOV_EXCL_LINE */
        }
        mem->data = (void*) (mem->buffer);
#else
        *status = OSKAR_ERR_OPENCL_NOT_AVAILABLE;         /* LCOV_EXCL_LINE */
#endif
    }
    else
    {
        *status = OSKAR_ERR_BAD_LOCATION;                 /* LCOV_EXCL_LINE */
    }

    /* Return a handle to the structure .*/
    return mem;
}

#ifdef __cplusplus
}
#endif
