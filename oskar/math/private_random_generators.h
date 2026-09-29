#ifndef OSKAR_PRIVATE_RANDOM_GENERATORS_H_
#define OSKAR_PRIVATE_RANDOM_GENERATORS_H_

/* Random123 marks its generators __device__ only when it recognises nvcc
 * (via __CUDACC__). Under hipcc it takes its generic clang/gcc path and
 * leaves them host-only, so kernels cannot call philox. It honours a
 * pre-set R123_CUDA_DEVICE, which avoids patching the bundled library. */
#if defined(__HIPCC__) || defined(__HIP__)
#ifndef R123_CUDA_DEVICE
#define R123_CUDA_DEVICE __host__ __device__
#endif
/* The generic path also enables SSE on an x86 host, which pulls in
 * features/sse.h and its host-only forward declaration of assemble_from_u32
 * -- a conflicting overload once R123_CUDA_DEVICE is set. SSE is meaningless
 * in device code; nvccfeatures.h turns it off for exactly this reason. */
#ifndef R123_USE_SSE
#define R123_USE_SSE 0
#endif
/* The generic path's R123_ASSERT is the C library assert(), which is
 * host-only. nvccfeatures.h traps instead; do the same, on either side. */
#ifndef R123_ASSERT
#define R123_ASSERT(x) do { if (!(x)) __builtin_trap(); } while (0)
#endif
#endif
#include <Random123/philox.h>

/* Use 32-bit integers for both single and double floating-point precision to
 * preserve random sequences. */
/* Generate two random integers. */
#define OSKAR_R123_GENERATE_2(S,C0,C1) \
        philox2x32_key_t k;     \
        philox2x32_ctr_t c;     \
        union {                 \
            philox2x32_ctr_t c; \
            uint32_t i[2];      \
        } u;                    \
        k.v[0] = S;             \
        c.v[0] = C0;            \
        c.v[1] = C1;            \
        u.c = philox2x32(c, k);

/* Generate four random integers. */
#define OSKAR_R123_GENERATE_4(S,C0,C1,C2,C3) \
        philox4x32_key_t k;     \
        philox4x32_ctr_t c;     \
        union {                 \
            philox4x32_ctr_t c; \
            uint32_t i[4];      \
        } u;                    \
        k.v[0] = S;             \
        k.v[1] = 0xCAFEF00DuL;  \
        c.v[0] = C0;            \
        c.v[1] = C1;            \
        c.v[2] = C2;            \
        c.v[3] = C3;            \
        u.c = philox4x32(c, k);

#endif
