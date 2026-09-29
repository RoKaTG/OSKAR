/* Copyright (c) 2018-2026, The OSKAR Developers. See LICENSE file. */

#ifndef M_CAT
#define M_CAT(A, B) M_CAT_(A, B)
#define M_CAT_(A, B) A##B
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

#if defined(_MSC_VER)
#define DO_PRAGMA(X) __pragma(X)\

#elif (__STDC_VERSION__ >= 199901L) || (__cplusplus >= 201103L)
#define DO_PRAGMA(X) _Pragma (#X)\

#else
#define DO_PRAGMA(X)
#endif

#define ATOMIC_ADD_CAPTURE(TYPE, ARRAY, IDX, VAL, OLD)\
    M_CAT(ATOMIC_ADD_CAPTURE_, TYPE)(ARRAY, IDX, VAL, OLD)
#define ATOMIC_ADD_UPDATE(TYPE, ARRAY, IDX, VAL)\
    M_CAT(ATOMIC_ADD_UPDATE_, TYPE)(ARRAY, IDX, VAL)
#define ROUND(FP, X) M_CAT(ROUND_, FP)(X)

#if defined(__CUDACC__) || defined(__HIPCC__) || defined(__HIP__)

/* == CUDA / HIP ========================================================== */
/* The two runtimes are source-compatible for everything below, so they share
 * one branch. What differs is how each advertises capability:
 *
 *   - NVIDIA gates on __CUDA_ARCH__, which hipcc does not define. An
 *     undefined identifier evaluates to 0 in #if, so a naive reuse of those
 *     guards silently selects the *fallback* definitions -- and the fallback
 *     for WARP_REDUCE is empty, which would drop most of every reduction
 *     without any diagnostic. The two OSKAR_GPU_HAS_* flags below exist so
 *     that trap cannot be re-introduced by editing a guard.
 *
 *   - AMD's wavefront is 64 lanes on CDNA and 32 on RDNA. See the note above
 *     WARP_REDUCE for why that does not change the reduction. */
#if defined(__HIPCC__) || defined(__HIP__)
    /* Every GPU HIP targets has cross-lane shuffles, and FP64 atomicAdd is
     * available on all supported architectures (lowered to a CAS loop by the
     * compiler where there is no hardware instruction, which is correct
     * either way). */
    #define OSKAR_GPU_HAS_SHUFFLE 1
    #define OSKAR_GPU_HAS_FP64_ATOMICS 1

    /* sincos() and rsqrt() are not <cmath> functions, so HIP does not get the
     * usual set of C++ float overloads for them: it declares sincos() for
     * double only, and the float instantiation of every kernel fails to
     * resolve. CUDA happens to supply both. Rather than depend on which
     * overloads a given ROCm release provides, dispatch explicitly to the
     * float and double entry points, which are stable. */
    __device__ static inline void oskar_sincos_(
            float x, float* s, float* c) { sincosf(x, s, c); }
    __device__ static inline void oskar_sincos_(
            double x, double* s, double* c) { sincos(x, s, c); }
    __device__ static inline float oskar_rsqrt_(float x) { return rsqrtf(x); }
    __device__ static inline double oskar_rsqrt_(double x) { return rsqrt(x); }
#else
    #if __CUDA_ARCH__ >= 300
        #define OSKAR_GPU_HAS_SHUFFLE 1
    #endif
    #if __CUDA_ARCH__ >= 600
        #define OSKAR_GPU_HAS_FP64_ATOMICS 1
    #endif
#endif

#define ATOMIC_ADD_CAPTURE_float(ARRAY, IDX, VAL, OLD)\
    OLD = atomicAdd(&ARRAY[IDX], VAL);
#define ATOMIC_ADD_CAPTURE_int(ARRAY, IDX, VAL, OLD)\
    OLD = atomicAdd(&ARRAY[IDX], VAL);
#define ATOMIC_ADD_UPDATE_float(ARRAY, IDX, VAL) atomicAdd(&ARRAY[IDX], VAL);
#define ATOMIC_ADD_UPDATE_int(  ARRAY, IDX, VAL) atomicAdd(&ARRAY[IDX], VAL);
#define BARRIER __syncthreads()
#define BARRIER_GLOBAL __syncthreads()
#define DEVICE_FUNC __device__
#define FMA(X, Y, Z) fma(X, Y, Z)
#define GLOBAL
#define GLOBAL_IN( TYPE, NAME) const TYPE *const __restrict__ NAME
#define GLOBAL_OUT(TYPE, NAME) TYPE *__restrict__ NAME
#define GLOBAL_ID_X (blockDim.x * blockIdx.x + threadIdx.x)
#define GLOBAL_ID_Y (blockDim.y * blockIdx.y + threadIdx.y)
#define GROUP_ID_X blockIdx.x
#define GROUP_ID_Y blockIdx.y
#define INLINE __device__ __forceinline__
#define KERNEL(NAME) __global__ void NAME
#define KERNEL_PUB(NAME) __global__ void NAME
#define KERNEL_LOOP_X(TYPE, I, OFFSET, N)\
    const TYPE I = blockDim.x * blockIdx.x + threadIdx.x + OFFSET;\
    if (I >= N) return;\

#define KERNEL_LOOP_Y(TYPE, I, OFFSET, N)\
    const TYPE I = blockDim.y * blockIdx.y + threadIdx.y + OFFSET;\
    if (I >= N) return;\

#define KERNEL_LOOP_PAR_X(TYPE, I, OFFSET, N) KERNEL_LOOP_X(TYPE, I, OFFSET, N)
#define KERNEL_LOOP_END \

#define LOCAL __shared__
#define LOCAL_CL(TYPE, NAME)
#define LOCAL_CUDA(X) X
#define LOCAL_CUDA_BASE(TYPE, NAME)\
    extern __shared__ __align__(64) unsigned char my_smem[];\
    TYPE* NAME = reinterpret_cast<TYPE*>(my_smem);\

#define LOCAL_DIM_X blockDim.x
#define LOCAL_DIM_Y blockDim.y
#define LOCAL_ID_X threadIdx.x
#define LOCAL_ID_Y threadIdx.y
#define LOOP_UNROLL DO_PRAGMA(unroll)
#define MAKE_ZERO(FP, X) X = (FP)0
#define MAKE_ZERO2(FP, X) X.x = X.y = (FP)0
#define MUTEX_LOCK(MUTEX) while (atomicCAS((MUTEX), 0, 1) != 0)
#define MUTEX_UNLOCK(MUTEX) atomicExch((MUTEX), 0)
#define OSKAR_REGISTER_KERNEL(NAME) OSKAR_CUDA_KERNEL(NAME)
#define ROUND_float(X) __float2int_rn(X)
#define ROUND_double(X) __double2int_rn(X)
#if defined(__HIPCC__) || defined(__HIP__)
#define RSQRT(X) oskar_rsqrt_(X)
#define SINCOS(X, S, C) oskar_sincos_(X, &S, &C)
#else
#define RSQRT(X) rsqrt(X)
#define SINCOS(X, S, C) sincos(X, &S, &C)
#endif
#define THREADFENCE_BLOCK __threadfence_block()

#ifdef OSKAR_GPU_HAS_FP64_ATOMICS
/* Native atomics. */
#define ATOMIC_ADD_CAPTURE_double(ARRAY, IDX, VAL, OLD)\
    OLD = atomicAdd(&ARRAY[IDX], VAL);
#define ATOMIC_ADD_UPDATE_double(ARRAY, IDX, VAL) atomicAdd(&ARRAY[IDX], VAL);
#else
#define ATOMIC_ADD_CAPTURE_double(ARRAY, IDX, VAL, OLD) {\
    unsigned long long int* addr = (unsigned long long int*)(&ARRAY[IDX]);\
    unsigned long long int assumed, old_ = *addr;\
    do { assumed = old_; old_ = atomicCAS(addr, assumed,\
            __double_as_longlong(VAL + __longlong_as_double(assumed))); }\
    while (assumed != old_);\
    OLD = __longlong_as_double(old_);\
    }\

#define ATOMIC_ADD_UPDATE_double(ARRAY, IDX, VAL) {\
    unsigned long long int* addr = (unsigned long long int*)(&ARRAY[IDX]);\
    unsigned long long int assumed, old_ = *addr;\
    do { assumed = old_; old_ = atomicCAS(addr, assumed,\
            __double_as_longlong(VAL + __longlong_as_double(assumed))); }\
    while (assumed != old_);\
    }\

#endif /* OSKAR_GPU_HAS_FP64_ATOMICS */

/* OSKAR's kernels are written in terms of 32-lane groups: they derive a warp
 * index as (threadIdx.x >> 5) and a lane as (threadIdx.x & 31), and each
 * group handles a different baseline. OSKAR_WARP_WIDTH pins the shuffle
 * segment to that logical width so the grouping holds on any hardware
 * wavefront, including AMD's 64-lane CDNA wavefront where one wavefront
 * spans two of these groups. */
#define OSKAR_WARP_WIDTH 32

#ifdef OSKAR_GPU_HAS_SHUFFLE
    #if defined(__HIPCC__) || defined(__HIP__)
        #define WARP_SHUFFLE(    VAR, SRC_LANE)  __shfl(VAR, SRC_LANE, OSKAR_WARP_WIDTH)
        #define WARP_SHUFFLE_XOR(VAR, LANE_MASK) __shfl_xor(VAR, LANE_MASK, OSKAR_WARP_WIDTH)
    #elif CUDART_VERSION >= 9000
        #define WARP_SHUFFLE(    VAR, SRC_LANE)  __shfl_sync(0xFFFFFFFF, VAR, SRC_LANE, OSKAR_WARP_WIDTH)
        #define WARP_SHUFFLE_XOR(VAR, LANE_MASK) __shfl_xor_sync(0xFFFFFFFF, VAR, LANE_MASK, OSKAR_WARP_WIDTH)
    #else
        #define WARP_SHUFFLE(    VAR, SRC_LANE)  __shfl(VAR, SRC_LANE, OSKAR_WARP_WIDTH)
        #define WARP_SHUFFLE_XOR(VAR, LANE_MASK) __shfl_xor(VAR, LANE_MASK, OSKAR_WARP_WIDTH)
    #endif

    #define WARP_BROADCAST(VAR, SRC_LANE) VAR = WARP_SHUFFLE(VAR, SRC_LANE)
    #define WARP_DECL(X) X

    /* Butterfly reduction over OSKAR_WARP_WIDTH lanes: five XOR steps cover
     * masks 1..16, which is exactly 32 lanes.
     *
     * Do NOT add a sixth step for 64-wide wavefronts. The extra step would
     * fold lanes 0-31 together with lanes 32-63, and those two halves belong
     * to *different baselines* -- the result would be silently wrong. A
     * 64-lane wavefront is meant to run two of these reductions side by
     * side, which is what the pinned shuffle width above guarantees. */
    #define WARP_REDUCE(A) {\
            (A) += WARP_SHUFFLE_XOR((A), 1);\
            (A) += WARP_SHUFFLE_XOR((A), 2);\
            (A) += WARP_SHUFFLE_XOR((A), 4);\
            (A) += WARP_SHUFFLE_XOR((A), 8);\
            (A) += WARP_SHUFFLE_XOR((A), 16);}\

#else
    #define WARP_BROADCAST(VAR, SRC_LANE) __syncthreads()
    #define WARP_DECL(X) __shared__ X
    /* Reached on the nvcc *host* pass, where __CUDA_ARCH__ is undefined, so
     * this has to stay a no-op. The guard below is what makes the dangerous
     * case -- a device pass with no shuffles -- loud instead of silent. */
    #define WARP_REDUCE(A)
#endif /* OSKAR_GPU_HAS_SHUFFLE */

#if defined(__HIP_DEVICE_COMPILE__) && !defined(OSKAR_GPU_HAS_SHUFFLE)
#error "HIP device pass without OSKAR_GPU_HAS_SHUFFLE: WARP_REDUCE would \
expand to nothing and every warp reduction would silently return only lane \
zero's partial sum. Fix the capability guard above rather than this check."
#endif


#elif defined(__OPENCL_VERSION__)

/* == OpenCL ============================================================== */
#define ATOMIC_ADD_CAPTURE_int(ARRAY, IDX, VAL, OLD)\
    OLD = atomic_add(&ARRAY[IDX], VAL);
#define ATOMIC_ADD_UPDATE_float(ARRAY, IDX, VAL) {\
    volatile global float* addr = &ARRAY[IDX];\
    union { unsigned int u; float f; } old, assumed, tmp; old.f = *addr;\
    do { assumed.f = old.f; tmp.f = VAL + assumed.f;\
        old.u = atomic_cmpxchg((volatile global unsigned int*)addr,\
                assumed.u, tmp.u); } while (assumed.u != old.u);\
    }\

#define ATOMIC_ADD_UPDATE_double(ARRAY, IDX, VAL) {\
    volatile global double* addr = &ARRAY[IDX];\
    union { ulong u; double f; } old, assumed, tmp; old.f = *addr;\
    do { assumed.f = old.f; tmp.f = VAL + assumed.f;\
        /*old.u = atom_cmpxchg((volatile global ulong*)addr,\
                assumed.u, tmp.u);*/ } while (assumed.u != old.u);\
    }\

#define ATOMIC_ADD_UPDATE_int(ARRAY, IDX, VAL) atomic_add(&ARRAY[IDX], VAL);
#define BARRIER barrier(CLK_LOCAL_MEM_FENCE)
#define BARRIER_GLOBAL barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE)
#define DEVICE_FUNC
#define FMA(X, Y, Z) fma(X, Y, Z)
#define GLOBAL global
#define GLOBAL_IN( TYPE, NAME) global const TYPE *const restrict NAME
#define GLOBAL_OUT(TYPE, NAME) global TYPE *restrict NAME
#define GLOBAL_ID_X get_global_id(0)
#define GLOBAL_ID_Y get_global_id(1)
#define GROUP_ID_X get_group_id(0)
#define GROUP_ID_Y get_group_id(1)
#define INLINE inline
#define KERNEL(NAME) kernel void NAME
#define KERNEL_PUB(NAME) kernel void NAME
#define KERNEL_LOOP_X(TYPE, I, OFFSET, N)\
    const TYPE I = get_global_id(0) + OFFSET;\
    if (I >= N) return;\

#define KERNEL_LOOP_Y(TYPE, I, OFFSET, N)\
    const TYPE I = get_global_id(1) + OFFSET;\
    if (I >= N) return;\

#define KERNEL_LOOP_PAR_X(TYPE, I, OFFSET, N) KERNEL_LOOP_X(TYPE, I, OFFSET, N)
#define KERNEL_LOOP_END \

#define LOCAL local
#define LOCAL_CL(TYPE, NAME) , local TYPE *restrict NAME
#define LOCAL_CUDA(X)
#define LOCAL_CUDA_BASE(TYPE, NAME)
#define LOCAL_DIM_X get_local_size(0)
#define LOCAL_DIM_Y get_local_size(1)
#define LOCAL_ID_X get_local_id(0)
#define LOCAL_ID_Y get_local_id(1)
#define LOOP_UNROLL
#define MAKE_ZERO(FP, X) X = M_CAT(convert_, FP)((int)(0))
#define MAKE_ZERO2(FP, X) X = M_CAT(M_CAT(convert_, FP), 2)((int2)(0, 0))
#define MUTEX_LOCK(MUTEX) while (atomic_cmpxchg((MUTEX), 0, 1) != 0)
#define MUTEX_UNLOCK(MUTEX) atomic_xchg((MUTEX), 0)
#define OSKAR_REGISTER_KERNEL(NAME)
#define ROUND_float(X) (int)rint(X)
#define ROUND_double(X) (int)rint(X)
#define RSQRT(X) rsqrt(X)
#define SINCOS(X, S, C) S = sincos(X, &C)
#define THREADFENCE_BLOCK mem_fence()
#define WARP_BROADCAST(VAR, SRC_LANE) barrier(CLK_LOCAL_MEM_FENCE)
#define WARP_DECL(X) local X

#else

/* == C =================================================================== */
#if __STDC_VERSION__ >= 199901L && !defined(__cplusplus)
#include <tgmath.h>
#elif defined(__cplusplus) || defined(_MSC_VER)
#include <cmath>
#endif

#define ATOMIC_ADD_CAPTURE_double(ARRAY, IDX, VAL, OLD)\
    DO_PRAGMA(omp atomic capture) { OLD = ARRAY[IDX]; ARRAY[IDX] += VAL; }
#define ATOMIC_ADD_CAPTURE_float(ARRAY, IDX, VAL, OLD)\
    DO_PRAGMA(omp atomic capture) { OLD = ARRAY[IDX]; ARRAY[IDX] += VAL; }
#define ATOMIC_ADD_CAPTURE_int(ARRAY, IDX, VAL, OLD)\
    DO_PRAGMA(omp atomic capture) { OLD = ARRAY[IDX]; ARRAY[IDX] += VAL; }
#define ATOMIC_ADD_UPDATE_double(ARRAY, IDX, VAL)\
    DO_PRAGMA(omp atomic update) ARRAY[IDX] += VAL;
#define ATOMIC_ADD_UPDATE_float(ARRAY, IDX, VAL)\
    DO_PRAGMA(omp atomic update) ARRAY[IDX] += VAL;
#define ATOMIC_ADD_UPDATE_int(ARRAY, IDX, VAL)\
    DO_PRAGMA(omp atomic update) ARRAY[IDX] += VAL;

#define DEVICE_FUNC static
#define FMA(X, Y, Z) (X * Y + Z)
#define GLOBAL
#define GLOBAL_IN( TYPE, NAME) const TYPE *const RESTRICT NAME
#define GLOBAL_OUT(TYPE, NAME) TYPE *RESTRICT NAME
#if __STDC_VERSION__ >= 199901L || defined(__cplusplus)
#define INLINE static inline
#else
#define INLINE static
#endif
#define KERNEL(NAME) static void NAME
#define KERNEL_PUB(NAME) void NAME
#define KERNEL_LOOP_X(TYPE, I, OFFSET, N)\
    TYPE I;\
    for (I = OFFSET; I < N; I++) {\

#define KERNEL_LOOP_Y(TYPE, I, OFFSET, N)\
    TYPE I;\
    for (I = OFFSET; I < N; I++) {\

#define KERNEL_LOOP_PAR_X(TYPE, I, OFFSET, N)\
    TYPE I;\
    DO_PRAGMA(omp parallel for private(I))\
    for (I = OFFSET; I < N; I++) {\

#define KERNEL_LOOP_END }\

#define LOCAL
#define LOCAL_CL(TYPE, NAME)
#define LOCAL_CUDA(X)
#define LOCAL_CUDA_BASE(TYPE, NAME)
#define MAKE_ZERO(FP, X) X = (FP)0
#define MAKE_ZERO2(FP, X) X.x = X.y = (FP)0
#define OSKAR_REGISTER_KERNEL(NAME)
#define ROUND_float(X) (int)roundf(X)
#define ROUND_double(X) (int)round(X)
#define RSQRT(X) (1 / sqrt(X))
#define SINCOS(X, S, C) S = sin(X); C = cos(X)
#define THREADFENCE_BLOCK

#endif

/* == General ============================================================= */
#if defined(__CUDACC__) && __CUDA_ARCH__ >= 350
#define OSKAR_LOAD_MATRIX(M, IND4c) {\
        M.a = __ldg(&(IND4c.a));\
        M.b = __ldg(&(IND4c.b));\
        M.c = __ldg(&(IND4c.c));\
        M.d = __ldg(&(IND4c.d));}\

#else
#define OSKAR_LOAD_MATRIX(M, IND4c) M = IND4c;\

#endif

#define OSKAR_SINC(FP, X) ( (X) == (FP)0 ? (FP)1 : (sin((FP) (X)) / (X)) )
