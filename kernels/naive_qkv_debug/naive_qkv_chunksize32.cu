#include "kittens.cuh"
// #include "pyutils/pyutils.cuh"
#include <fstream>
#include <chrono>
#include <cmath>

#define NUM_WARPS 8
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

#ifndef ATTN_B
constexpr int ATTN_B = 16; // batch size
#endif

#ifndef ATTN_H
constexpr int ATTN_H = 8;  // number of heads
#endif

#ifndef ATTN_D
constexpr int ATTN_D = 128; // dimension
#endif

#ifndef ATTN_F
constexpr int ATTN_F = 128;  // number of features
#endif

#ifndef ATTN_N
constexpr int ATTN_N = 1024; // sequence length
#endif

constexpr int CHUNK_SIZE = 32;


// debug
// #ifndef ATTN_B
// constexpr int ATTN_B = 1;//16; // batch size
// #endif
// #ifndef ATTN_H
// constexpr int ATTN_H = 1;  // number of heads
// #endif
// #ifndef ATTN_D
// constexpr int ATTN_D = 128; // dimension
// #endif
// #ifndef ATTN_F
// constexpr int ATTN_F = 128;  // number of features
// #endif
// #ifndef ATTN_N
// constexpr int ATTN_N = 1024;//32;//64;//1024; // sequence length
// #endif
// constexpr int CHUNK_SIZE = 32;


constexpr int V_CHUNK_SIZE = 32;

using namespace kittens;

using G = kittens::group<NUM_WARPS>;

#define BARRIER { \
    asm volatile("s_waitcnt vmcnt(0)"); \
    asm volatile("s_waitcnt lgkmcnt(0)"); \
    __builtin_amdgcn_s_barrier(); \
    __builtin_amdgcn_sched_barrier(0); \
    __syncthreads(); \
}

__device__ bool thread0() {
    return threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0;
}
__device__ bool thread(int tid) {
    return threadIdx.x == tid && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0;
}

#define D(x) do { if(thread0())printf("%d,  " #x ": %lf\n",  __LINE__, static_cast<float>(x)); } while (0)
#define Dk(x) do { if(thread0())printf("%d, K:%d " #x ": %lf\n",  __LINE__, k, static_cast<float>(x)); } while (0)
//  #define D(x) 
//  #define Dk(x) 
#define Dk2(x) do { if(thread0())printf("%d, K:%d " #x ": %lf\n",  __LINE__, k, static_cast<float>(x)); } while (0)


#define PT(xx) \
    if(thread0()) \
        for(int i = 0; i < xx.height; i++) \
            for(int j = 0; j < xx.width; j++) \
                for(int k = 0; k < xx.packed_per_base_tile; k++) {\
                    printf("%d, " #xx ".[%d][%d].data[%d] %lf\n",  __LINE__,i,j,k, float(xx.tiles[i][j].data[k].x)); \
                    printf("%d, " #xx ".[%d][%d].data[%d] %lf\n",  __LINE__,i,j,k, float(xx.tiles[i][j].data[k].y)); \
                }\

template<typename T, typename D>
__device__ T sum_tile(const D& A2_tile){ 
    T sum_val_A = T(0);

    #pragma unroll
    for(int i = 0; i < D::height; i++){
        #pragma unroll
        for(int j = 0; j < D::width; j++){
            #pragma unroll
            for(int k = 0; k < D::packed_per_base_tile; k++){
                sum_val_A += static_cast<T>(A2_tile.tiles[i][j].data[k].x);
                sum_val_A += static_cast<T>(A2_tile.tiles[i][j].data[k].y);
            }
        }
    }
    return sum_val_A;
}

// template<int ATTN_F, typename T=bf16, typename L=row_l, typename S=rt_32x16_s> using q_tile = rt<T, CHUNK_SIZE, ATTN_F, L, S>;
// template<int ATTN_F, typename T=bf16, typename L=col_l, typename S=rt_16x32_s> using q_tile_transposed = rt<T, ATTN_F, CHUNK_SIZE, L, S>;
// template<int ATTN_F, typename T=bf16, typename L=row_l, typename S=rt_32x16_s> using k_tile = rt<T, CHUNK_SIZE, ATTN_F, L, S>;
// template<int ATTN_D, typename T=bf16, typename L=row_l, typename S=rt_32x16_s> using v_tile = rt<T, CHUNK_SIZE, ATTN_D, L, S>;
// template<int ATTN_F, typename T=bf16, typename L=col_l, typename S=rt_16x32_s> using k_tile_transposed = rt<T, ATTN_F, CHUNK_SIZE, L, S>;
// template<int ATTN_D, typename T=float, typename L=col_l, typename S=rt_16x32_4_s> using attn_tile = rt<T, CHUNK_SIZE, CHUNK_SIZE, L, S>;
// template<int ATTN_D, typename T=bf16, typename L=row_l, typename S=rt_32x16_s> using o_tile = rt<T, CHUNK_SIZE, ATTN_D, L, S>;
// template<int ATTN_D, typename T=bf16, typename L=col_l, typename S=rt_16x32_s> using o_tile_transposed = rt<T, ATTN_D, CHUNK_SIZE, L, S>;
template<int F, typename T=bf16, typename L=row_l, typename S=rt_32x16_s> using q_tile = rt<T, CHUNK_SIZE, F, L, S>;
template<int F, typename T=bf16, typename L=col_l, typename S=rt_16x32_s> using q_tile_transposed = rt<T, F, CHUNK_SIZE, L, S>;
template<int F, typename T=bf16, typename L=row_l, typename S=rt_32x16_s> using k_tile = rt<T, CHUNK_SIZE, F, L, S>;
// template<int D, typename T=bf16, typename L=row_l, typename S=rt_32x16_s> using v_tile = rt<T, CHUNK_SIZE, D, L, S>;
template<int D, typename T=bf16, typename L=row_l, typename S=rt_32x16_s> using v_tile = rt<T, CHUNK_SIZE, V_CHUNK_SIZE, L, S>;
template<int F, typename T=bf16, typename L=col_l, typename S=rt_16x32_s> using k_tile_transposed = rt<T, F, CHUNK_SIZE, L, S>;
template<int D, typename T=float, typename L=col_l, typename S=rt_16x32_4_s> using attn_tile = rt<T, CHUNK_SIZE, CHUNK_SIZE, L, S>;
template<int D, typename T=float, typename L=row_l, typename S=rt_32x16_4_s> using attn_tile_transposed = rt<T, CHUNK_SIZE, CHUNK_SIZE, L, S>;
// template<int D, typename T=bf16, typename L=row_l, typename S=rt_32x16_s> using o_tile = rt<T, CHUNK_SIZE, D, L, S>;
// template<int D, typename T=bf16, typename L=col_l, typename S=rt_16x32_s> using o_tile_transposed = rt<T, D, CHUNK_SIZE, L, S>;
template<int D, typename T=bf16, typename L=row_l, typename S=rt_32x16_s> using o_tile = rt<T, CHUNK_SIZE, V_CHUNK_SIZE, L, S>;
template<int D, typename T=bf16, typename L=col_l, typename S=rt_16x32_s> using o_tile_transposed = rt<T, V_CHUNK_SIZE, CHUNK_SIZE, L, S>;

using _gl_QKVO = gl<bf16, -1, -1, -1, -1>;

struct lightning_attn2_globals {
    // // shapes    
    // using q_tile       = st_bf<CHUNK_SIZE, ATTN_F>;
    // using k_tile       = st_bf<CHUNK_SIZE, ATTN_F>;
    // using k_tile_split = st_bf<CHUNK_SIZE, ATTN_F/2>;
    // using v_tile       = st_bf<CHUNK_SIZE, ATTN_D>;
    // using o_tile       = st_bf<CHUNK_SIZE, ATTN_D>;

    // // global layouts
    // using q_gl       = gl<bf16,  -1, -1, -1, -1, q_tile>; // TODO: gl<bf16,  -1, -1, -1, -1>; No TMA types
    // using k_gl       = gl<bf16,  -1, -1, -1, -1, k_tile>;
    // using k_split_gl = gl<bf16,  -1, -1, -1, -1, k_tile_split>;
    // using v_gl       = gl<bf16,  -1, -1, -1, -1, v_tile>;
    // using o_gl       = gl<bf16,  -1, -1, -1, -1, o_tile>;

    // shapes    
    // using q_tile       = st_bf<CHUNK_SIZE, ATTN_F>;
    // using k_tile       = st_bf<CHUNK_SIZE, ATTN_F>;
    // using k_tile_split = st_bf<CHUNK_SIZE, ATTN_F/2>;
    // using v_tile       = st_bf<CHUNK_SIZE, ATTN_D>;
    // using o_tile       = st_bf<CHUNK_SIZE, ATTN_D>;

    // global layouts
    // using q_gl       = gl<bf16,  -1, -1, -1, -1>; // TODO: gl<bf16,  -1, -1, -1, -1>; No TMA types
    // using k_gl       = gl<bf16,  -1, -1, -1, -1>;
    // using k_split_gl = gl<bf16,  -1, -1, -1, -1>;
    // using v_gl       = gl<bf16,  -1, -1, -1, -1>;
    // using o_gl       = gl<bf16,  -1, -1, -1, -1>;

    // q_gl Qg;
    // k_gl Kg;
    // k_split_gl K_split_g;
    // v_gl Vg;
    // o_gl Og;

    _gl_QKVO Qg, Kg, K_split_g, Vg, Og;
    _gl_QKVO ODEBUGg;

    // float *slopes;
    uintptr_t slopes;

    hipStream_t stream;
    // dim3 grid() {return dim3(ATTN_B, ATTN_H);}
    // dim3 block() {return dim3(ATTN_F, 1, 1);}
    dim3 block() {return dim3(NUM_THREADS);}
    // dim3 grid() { return dim3(ATTN_H, ((ATTN_N / CHUNK_SIZE + NUM_WARPS - 1) / NUM_WARPS), ATTN_B); }
    // dim3 grid() {return dim3(ATTN_H, ATTN_B);}
    // dim3 grid() {return dim3(ATTN_H, ATTN_B, ATTN_D / V_CHUNK_SIZE);} // parallel in V's ATTN_D dim
    dim3 grid() {return dim3(ATTN_D / V_CHUNK_SIZE, ATTN_H, ATTN_B);} // parallel in V's ATTN_D dim
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

__global__ __launch_bounds__(NUM_THREADS, 2)
void qkv_kernel(const lightning_attn2_globals globals, int N)
{
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    // smem
    st_bf<CHUNK_SIZE, ATTN_F, st_32x32_s> (&q_smem)[2] = al.allocate<st_bf<CHUNK_SIZE, ATTN_F, st_32x32_s>, 2>();                   // 64x128 x2 x2 = 32768 B = 32 KB
    st_bf<CHUNK_SIZE, ATTN_F, st_32x32_s> (&k_smem)[2] = al.allocate<st_bf<CHUNK_SIZE, ATTN_F, st_32x32_s>, 2>();                   // 64x128 x2 x2 = 32768 B = 32 KB
    // st_bf<CHUNK_SIZE, ATTN_D, st_32x32_s> (&v_smem)[2] = al.allocate<st_bf<CHUNK_SIZE, ATTN_D, st_32x32_s>, 2>();
    st_bf<CHUNK_SIZE, V_CHUNK_SIZE, st_32x32_s> (&v_smem)[2] = al.allocate<st_bf<CHUNK_SIZE, V_CHUNK_SIZE, st_32x32_s>, 2>();    // 64x32 x2 x2 = 8192 B = 8 KB
    // st_bf<ATTN_F, ATTN_D, st_32x32_s> (&kv_state_smem) = al.allocate<st_bf<ATTN_F, ATTN_D, st_32x32_s>>();

    // col_vec<st_fl<CHUNK_SIZE, ATTN_D, st_32x32_s>> (&q_decay) = al.allocate<col_vec<st_fl<CHUNK_SIZE, ATTN_D, st_32x32_s>>>();
    // col_vec<st_fl<CHUNK_SIZE, ATTN_D, st_32x32_s>> (&k_decay) = al.allocate<col_vec<st_fl<CHUNK_SIZE, ATTN_D, st_32x32_s>>>();

    // const int head_idx = blockIdx.x;
    // const int batch_idx = blockIdx.y;
    // const int v_block_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int batch_idx = blockIdx.z;
    const int v_block_idx = blockIdx.x;

    // printf("head_idx %d batch_idx %d\n", head_idx, batch_idx);
    // float slope = globals.slopes[head_idx];
    float slope = reinterpret_cast<float*>(globals.slopes)[head_idx];

    int blocks = N / CHUNK_SIZE;
    // printf("blocks: %d\n", blocks);

    int tic = 0, toc = 1;

    // Initialize all of the register tiles.
    q_tile<ATTN_F, bf16> q_reg;                         // [CHUNK_SIZE, ATTN_F], 32x128
    q_tile_transposed<ATTN_F, bf16> q_reg_transposed;   // [ATTN_F, CHUNK_SIZE], 128x32, rt_16x32_s
    k_tile<ATTN_F, bf16> k_reg;                         // [CHUNK_SIZE, ATTN_F], 32x128
    k_tile_transposed<ATTN_F, bf16> k_reg_transposed;   // [ATTN_F, CHUNK_SIZE], 128x32, rt_16x32_s
    v_tile<ATTN_D, bf16, col_l, rt_32x32_s> v_reg;                    // [CHUNK_SIZE, V_CHUNK_SIZE], 32x32
    o_tile_transposed<ATTN_D, float, col_l, rt_32x32_s> o_reg;          // [V_CHUNK_SIZE, CHUNK_SIZE], 32x32
    // o_tile_transposed<CHUNK_SIZE, float, col_l, rt_32x32_s> o_reg;          // [CHUNK_SIZE, CHUNK_SIZE], 64x64
    attn_tile<ATTN_D, float, col_l, rt_32x32_s> attn_block[2];          // [CHUNK_SIZE, CHUNK_SIZE], 32x32, 1x1个subtiles, rt_32x32_s
    attn_tile<ATTN_D, bf16, col_l, rt_32x32_s> attn_block_bf16;         // [CHUNK_SIZE, CHUNK_SIZE], 32x32, 1x1个subtiles, rt_32x32_s
    attn_tile<ATTN_D, bf16, col_l, rt_16x32_4_s> attn_block_bf16_in;    // [64x64], 内部16x32，4x2个subtiles
    // attn_tile<ATTN_D, bf16, col_l, rt_32x32_s> attn_block_bf16_in;    // [64x64], 内部16x32，4x2个subtiles 

    zero(o_reg);

    // using T = typename q_tile<ATTN_F, bf16>::dtype; // 32x16
    
    // using st_q_tile              = st_bf<CHUNK_SIZE, ATTN_F, st_32x32_s>;      // 64 x 128
    // using T = st_q_tile::dtype;
    using T = typename st_bf<CHUNK_SIZE, ATTN_F, st_32x32_s>::dtype;
    constexpr int bytes_per_thread = st_32x32_s::template bytes_per_thread<T>();
    constexpr int bytes_per_memcpy = bytes_per_thread * NUM_THREADS;
    constexpr int memcpy_per_tile_q_k = CHUNK_SIZE * ATTN_F * sizeof(T) / bytes_per_memcpy;
    // constexpr int memcpy_per_tile_v = CHUNK_SIZE * ATTN_D * sizeof(T) / bytes_per_memcpy;
    constexpr int memcpy_per_tile_v = CHUNK_SIZE * V_CHUNK_SIZE * sizeof(T) / bytes_per_memcpy;
    uint32_t swizzled_offsets_Q[memcpy_per_tile_q_k];
    uint32_t swizzled_offsets_V[memcpy_per_tile_v];
    uint32_t swizzled_offsets_K[memcpy_per_tile_q_k];
    G::prefill_swizzled_offsets<1, false>(q_smem[0], globals.Qg, swizzled_offsets_Q);
    G::prefill_swizzled_offsets<1, false>(k_smem[0], globals.Kg, swizzled_offsets_K);
    G::prefill_swizzled_offsets<1, false>(v_smem[0], globals.Vg, swizzled_offsets_V);


    for (int block = 0; block < blocks; block++) {
        zero(o_reg);
        // Load Q, K, V tiles from global memory to shared memory
        G::load<1, false>(q_smem[tic], globals.Qg, {batch_idx, block, head_idx, 0}, swizzled_offsets_Q);
        G::load<1, false>(k_smem[tic], globals.Kg, {batch_idx, block, head_idx, 0}, swizzled_offsets_K);
        G::load<1, false>(v_smem[tic], globals.Vg, {batch_idx, block, head_idx, v_block_idx}, swizzled_offsets_V);
        // G::load<1, false>(v_smem[tic], globals.Vg, {batch_idx, block, head_idx, v_block_idx});
        BARRIER;

        // smem to reg
        load(q_reg, q_smem[tic]);
        load(k_reg, k_smem[tic]);
        load(v_reg, v_smem[tic]);
        BARRIER;

        // calculation QK
        zero(attn_block[tic]);
        transpose(q_reg_transposed, q_reg);
        transpose(k_reg_transposed, k_reg);
        mma_AtB(attn_block[0], k_reg_transposed, q_reg_transposed, attn_block[0]);
        BARRIER;// need sync

#ifdef DEBUG
        // DEBUG dump result QK
        if (blockIdx.x == 0 && blockIdx.y == 0) {
            // printf("attn_block[0] height %d width %d\n", attn_block[0].height, attn_block[0].width);

            // tile[0][0]
            for (int i = 0; i < 4; i++) {
                if (threadIdx.x == 0 && threadIdx.y == 0) {
                    // float temp = (float)(attn_block[0].tiles[0][0].data[i]);
                    float temp = (attn_block[0].tiles[0][0].data[i])[0]; // HIP_vector_type<float, 2>
                    // printf("attn_block[0].data[%d]: %f\n", i, (attn_block[0].tiles[0][0].data[i]));
                    printf("attn_block[0].tiles[0][0].data[%d]: %f\n", i*2, temp);
                    temp = (attn_block[0].tiles[0][0].data[i])[1];
                    printf("attn_block[0].tiles[0][0].data[%d]: %f\n", i*2+1, temp);
                }
            }
            // int total_num_elements = attn_block[0]
            // for (int i = total_num_elements - 8; i < total_num_elements; i++) {
            
            // tile [0][1]
            for (int i = 0; i < 4; i++) {
                if (threadIdx.x == 0 && threadIdx.y == 0) {
                    float temp = (attn_block[0].tiles[0][1].data[i])[0]; // HIP_vector_type<float, 2>
                    printf("attn_block[0].tiles[0][1].data[%d]: %f\n", i*2, temp);
                    temp = (attn_block[0].tiles[0][1].data[i])[1];
                    printf("attn_block[0].tiles[0][1].data[%d]: %f\n", i*2+1, temp);
                }
            }
            // tile [1][0]
            for (int i = 0; i < 4; i++) {
                if (threadIdx.x == 0 && threadIdx.y == 0) {
                    float temp = (attn_block[0].tiles[1][0].data[i])[0]; // HIP_vector_type<float, 2>
                    printf("attn_block[0].tiles[1][0].data[%d]: %f\n", i*2, temp);
                    temp = (attn_block[0].tiles[1][0].data[i])[1];
                    printf("attn_block[0].tiles[1][0].data[%d]: %f\n", i*2+1, temp);
                }
            }
            // tile [1][1]
            for (int i = 0; i < 4; i++) {
                if (threadIdx.x == 0 && threadIdx.y == 0) {
                    float temp = (attn_block[0].tiles[1][1].data[i])[0]; // HIP_vector_type<float, 2>
                    printf("attn_block[0].tiles[1][1].data[%d]: %f\n", i*2, temp);
                    temp = (attn_block[0].tiles[1][1].data[i])[1];
                    printf("attn_block[0].tiles[1][1].data[%d]: %f\n", i*2+1, temp);
                }
            }
        }
#endif

        // apply diag decay
        // TODO

        // how to copy 64x64 fp32 attn_block to 64x64 bf16 attn_block_bf16?
        // TODO
        // attn_block_bf16 [CHUNK_SIZE, CHUNK_SIZE], 64x64
        // copy(subtile_inplace<32>(attn_block_bf16, 0), subtile_inplace<32>(attn_block[0], 0));
        // copy(subtile_inplace<32>(attn_block_bf16, 1), subtile_inplace<32>(attn_block[0], 1));
        // copy(subtile_inplace<32>(attn_block_bf16, 0), subtile_inplace<32>(attn_block[0], 0));
        copy(attn_block_bf16, attn_block[0]);

#ifdef DEBUG
        // DEBUG dump result QK in bf16
        if (blockIdx.x == 0 && blockIdx.y == 0) {
            // printf("attn_block_bf16 height %d width %d\n", attn_block_bf16.height, attn_block_bf16.width);

            // tile[0][0]
            for (int i = 0; i < 4; i++) {
                if (threadIdx.x == 0 && threadIdx.y == 0) {
                    float temp = __bfloat162float((attn_block_bf16.tiles[0][0].data[i].x)); // HIP_vector_type<float, 2>
                    printf("attn_block_bf16.tiles[0][0].data[%d]: %f\n", i*2, temp);
                    temp = __bfloat162float((attn_block_bf16.tiles[0][0].data[i].y));
                    printf("attn_block_bf16.tiles[0][0].data[%d]: %f\n", i*2+1, temp);
                }
            }
            // tile [0][1]
            for (int i = 0; i < 4; i++) {
                if (threadIdx.x == 0 && threadIdx.y == 0) {
                    float temp = __bfloat162float((attn_block_bf16.tiles[0][1].data[i].x)); // HIP_vector_type<float, 2>
                    printf("attn_block_bf16.tiles[0][1].data[%d]: %f\n", i*2, temp);
                    temp = __bfloat162float((attn_block_bf16.tiles[0][1].data[i].y));
                    printf("attn_block_bf16.tiles[0][1].data[%d]: %f\n", i*2+1, temp);
                }
            }
            // tile [1][0]
            for (int i = 0; i < 4; i++) {
                if (threadIdx.x == 0 && threadIdx.y == 0) {
                    float temp = __bfloat162float((attn_block_bf16.tiles[1][0].data[i].x)); // HIP_vector_type<float, 2>
                    printf("attn_block_bf16.tiles[1][0].data[%d]: %f\n", i*2, temp);
                    temp = __bfloat162float((attn_block_bf16.tiles[1][0].data[i].y));
                    printf("attn_block_bf16.tiles[1][0].data[%d]: %f\n", i*2+1, temp);
                }
            }
            // tile [1][1]
            for (int i = 0; i < 4; i++) {
                if (threadIdx.x == 0 && threadIdx.y == 0) {
                    float temp = __bfloat162float((attn_block_bf16.tiles[1][1].data[i].x)); // HIP_vector_type<float, 2>
                    printf("attn_block_bf16.tiles[1][1].data[%d]: %f\n", i*2, temp);
                    temp = __bfloat162float((attn_block_bf16.tiles[1][1].data[i].y));
                    printf("attn_block_bf16.tiles[1][1].data[%d]: %f\n", i*2+1, temp);
                }
            }
        }
#endif
        // attn_block_bf16_in = *reinterpret_cast<attn_tile<ATTN_D, bf16, col_l, rt_16x32_4_s>*>(&attn_block_bf16);
        BARRIER;
#ifdef DEBUG
        // DEBUG dump result QK in bf16， 4x2 subtiles
        if (blockIdx.x == 0 && blockIdx.y == 0) {
            // printf("attn_block_bf16_in height %d width %d\n", attn_block_bf16_in.height, attn_block_bf16_in.width);

            // tile[0][0]
            for (int i = 0; i < 4; i++) {
                if (threadIdx.x == 0 && threadIdx.y == 0) {
                    float temp = __bfloat162float((attn_block_bf16_in.tiles[0][0].data[i].x)); // HIP_vector_type<float, 2>
                    printf("attn_block_bf16_in.tiles[0][0].data[%d]: %f\n", i*2, temp);
                    temp = __bfloat162float((attn_block_bf16_in.tiles[0][0].data[i].y));
                    printf("attn_block_bf16_in.tiles[0][0].data[%d]: %f\n", i*2+1, temp);
                }
            }
            // tile [0][1]
            for (int i = 0; i < 4; i++) {
                if (threadIdx.x == 0 && threadIdx.y == 0) {
                    float temp = __bfloat162float((attn_block_bf16_in.tiles[0][1].data[i].x)); // HIP_vector_type<float, 2>
                    printf("attn_block_bf16_in.tiles[0][1].data[%d]: %f\n", i*2, temp);
                    temp = __bfloat162float((attn_block_bf16_in.tiles[0][1].data[i].y));
                    printf("attn_block_bf16_in.tiles[0][1].data[%d]: %f\n", i*2+1, temp);
                }
            }
            // tile [1][0]
            for (int i = 0; i < 4; i++) {
                if (threadIdx.x == 0 && threadIdx.y == 0) {
                    // float temp = __bfloat162float((attn_block_bf16_in.tiles[1][0].data[i].x)); // HIP_vector_type<float, 2>
                    float temp = __bfloat162float((attn_block_bf16_in.tiles[2][0].data[i].x)); // HIP_vector_type<float, 2>
                    printf("attn_block_bf16_in.tiles[1][0].data[%d]: %f\n", i*2, temp);
                    // temp = __bfloat162float((attn_block_bf16_in.tiles[1][0].data[i].y));
                    temp = __bfloat162float((attn_block_bf16_in.tiles[2][0].data[i].y));
                    printf("attn_block_bf16_in.tiles[1][0].data[%d]: %f\n", i*2+1, temp);
                }
            }
            // tile [1][1]
            for (int i = 0; i < 4; i++) {
                if (threadIdx.x == 0 && threadIdx.y == 0) {
                    // float temp = __bfloat162float((attn_block_bf16_in.tiles[1][1].data[i].x)); // HIP_vector_type<float, 2>
                    float temp = __bfloat162float((attn_block_bf16_in.tiles[2][1].data[i].x)); // HIP_vector_type<float, 2>
                    printf("attn_block_bf16_in.tiles[1][1].data[%d]: %f\n", i*2, temp);
                    // temp = __bfloat162float((attn_block_bf16_in.tiles[1][1].data[i].y));
                    temp = __bfloat162float((attn_block_bf16_in.tiles[2][1].data[i].y));
                    printf("attn_block_bf16_in.tiles[1][1].data[%d]: %f\n", i*2+1, temp);
                }
            }
        }

        // DEBUG dump
        if (blockIdx.x == 0 && blockIdx.y == 0) {
            for (int i = 0; i < 8; i++) {
                // printf("v_smem[0].data[%d]: %f\n", i, (float)v_smem[0].data[i]);
                if (threadIdx.x == 0 && threadIdx.y == 0)
                    printf("q_smem[0].data[%d]: %f\n", i, __bfloat162float(q_smem[0].data[i]));
            }
            int total_num_elements = q_smem[0].rows * q_smem[0].cols;
            for (int i = total_num_elements - 8; i < total_num_elements; i++) {
                if (threadIdx.x == 0 && threadIdx.y == 0)
                    printf("q_smem[0].data[%d]: %f\n", i, __bfloat162float(q_smem[0].data[i]));
            }
        }
        if (blockIdx.x == 0 && blockIdx.y == 0) {
            for (int i = 0; i < 8; i++) {
                // printf("v_smem[0].data[%d]: %f\n", i, (float)v_smem[0].data[i]);
                if (threadIdx.x == 0 && threadIdx.y == 0)
                    printf("v_smem[0].data[%d]: %f\n", i, __bfloat162float(v_smem[0].data[i]));
            }
            int total_num_elements = v_smem[0].rows * v_smem[0].cols;
            for (int i = total_num_elements - 8; i < total_num_elements; i++) {
                if (threadIdx.x == 0 && threadIdx.y == 0)
                    printf("v_smem[0].data[%d]: %f\n", i, __bfloat162float(v_smem[0].data[i]));
            }
        }
#endif
        // calculate AV
        // v_reg [CHUNK_SIZE, ATTN_D], 32x128
        // o_reg [ATTN_D, CHUNK_SIZE], 128x32
        // v_reg * attn_block_bf16 -> o_reg, 32x128^T * 32x32 = 128x32
        mma_AtB(o_reg, v_reg, attn_block_bf16, o_reg); // subtile 32x32
        BARRIER;
        
        o_tile<ATTN_D, float, row_l, rt_32x32_s> o_reg_transposed; // [CHUNK_SIZE, V_CHUNK_SIZE]
        transpose(o_reg_transposed, o_reg);
        // store<1>(globals.Og, o_reg_transposed, {batch_idx, block, head_idx, 0});
        store<1>(globals.Og, o_reg_transposed, {batch_idx, block, head_idx, v_block_idx});

        // attn_tile<ATTN_D, float, row_l, rt_32x32_s> attn_block_trans;
        attn_tile_transposed<ATTN_D, float, row_l, rt_32x32_s> attn_block_trans;
        transpose(attn_block_trans, attn_block[0]);
        store<1>(globals.ODEBUGg, attn_block_trans, {batch_idx, block, head_idx, 0}); // for attn_block, no v split
        
    }

    // o_tile<ATTN_D, float, row_l, rt_32x32_s> o_reg_transposed;
    // transpose(o_reg_transposed, o_reg);
    // store<1>(globals.Og, o_reg_transposed, {batch_idx, 0, head_idx, 0});
}

lightning_attn2_globals lightning_attn2_init(
    bf16 *d_q, bf16 *d_k, bf16 *d_v, bf16 *d_o,
    // debug dump output,
    bf16 *d_debug_o,
    float *d_slopes,
    // int B, int H, int N, int ATTN_F, int ATTN_D
    unsigned long B, unsigned long H, unsigned long N
) {
    // global pointers. 
    // using q_tile       = st_bf<CHUNK_SIZE,   ATTN_F>;
    // using k_tile       = st_bf<CHUNK_SIZE,   ATTN_F>;
    // using k_split_tile = st_bf<CHUNK_SIZE,   ATTN_F/2>;
    // using v_tile       = st_bf<CHUNK_SIZE,   ATTN_D>;
    // using o_tile       = st_bf<CHUNK_SIZE,   ATTN_D>;
    
    // using q_global       = gl<bf16, -1, -1, -1, -1, q_tile>;
    // using k_global       = gl<bf16, -1, -1, -1, -1, k_tile>;
    // using k_split_global = gl<bf16, -1, -1, -1, -1, k_split_tile>;
    // using v_global       = gl<bf16, -1, -1, -1, -1, v_tile>;
    // using o_global       = gl<bf16, -1, -1, -1, -1, o_tile>;

    using globals = lightning_attn2_globals;
    // q_global             q_arg{d_q, B, H, N, ATTN_F};
    // k_global             k_arg{d_k, B, H, N, ATTN_F};
    // k_split_global k_split_arg{d_k, B, H, N, ATTN_F}; 
    // v_global             v_arg{d_v, B, H, N, ATTN_D};
    // o_global             o_arg{d_o, B, H, N, ATTN_D};

    // _gl_QKVO             q_arg{d_q, B, H, N, ATTN_F};
    // _gl_QKVO             k_arg{d_k, B, H, N, ATTN_F};
    // _gl_QKVO             k_split_arg{d_k, B, H, N, ATTN_F}; 
    // _gl_QKVO             v_arg{d_v, B, H, N, ATTN_D};
    // _gl_QKVO             o_arg{d_o, B, H, N, ATTN_D};
    _gl_QKVO             q_arg{d_q, B, N, H, ATTN_F};
    _gl_QKVO             k_arg{d_k, B, N, H, ATTN_F};
    _gl_QKVO             k_split_arg{d_k, B, N, H, ATTN_F}; 
    _gl_QKVO             v_arg{d_v, B, N, H, ATTN_D};
    _gl_QKVO             o_arg{d_o, B, N, H, ATTN_D};

    // debug dump
    _gl_QKVO            odebug_arg{d_debug_o, B, N, H, CHUNK_SIZE};

    globals g{
        q_arg, k_arg, k_split_arg, v_arg, o_arg,
        // debug dump
        odebug_arg,
        // d_slopes
        reinterpret_cast<uintptr_t>(d_slopes)
    };

    return g;
}

// void dispatch_micro(lightning_attn2_globals g) {
//     //temp test
//     float* dev_ptr = reinterpret_cast<float*>(g.slopes);

//     unsigned long mem_size = g.dynamic_shared_memory();
//     hipFuncSetAttribute((void*)lightning_attn2_kernel, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
//     lightning_attn2_kernel<<<g.grid(), g.block(), mem_size, g.stream>>>(g, ATTN_N);
// }

// PYBIND11_MODULE(tk_kernel, m) {
//     m.doc() = "tk_kernel python module";
//     py::bind_function<dispatch_micro>(m, "dispatch_micro", 
//         &lightning_attn2_globals::Qg, 
//         &lightning_attn2_globals::Kg,
//         &lightning_attn2_globals::K_split_g, 
//         &lightning_attn2_globals::Vg, 
//         &lightning_attn2_globals::Og,
//         &lightning_attn2_globals::slopes
//     );
// }

#define HipCheckError()    __hipCheckError( __FILE__, __LINE__ )
inline void __hipCheckError( const char *file, const int line ) {
    hipError_t err = hipGetLastError();
    if ( hipSuccess != err ) {
        fprintf( stderr, "hipCheckError() failed at %s:%i : %s\n",
                 file, line, hipGetErrorString( err ) );
        exit( -1 );
    }
    err = hipDeviceSynchronize();
    if( hipSuccess != err ) {
        fprintf( stderr, "hipCheckError() with sync failed at %s:%i : %s\n",
                 file, line, hipGetErrorString( err ) );
        exit( -1 );
    }
}

int main(int argc, char **argv) {
    constexpr int B = 16;
    constexpr int D = 128;
    constexpr int H = 8;
    constexpr int F = 128;
    constexpr int N = 1024;

    // constexpr int B = 1;// 16;
    // constexpr int D = 128;
    // constexpr int H = 1;//8;
    // constexpr int F = 128;
    // constexpr int N = 1024;//32;//64;

    constexpr int warmup_iters = 1;
    constexpr int timing_iters = 1;

    int TOTAL_ELEMENTS_QK = B * H * N * D;
    int TOTAL_ELEMENTS_VO = B * H * N * F;

    float *slopes      = new float[H];
    float *q           = new float[TOTAL_ELEMENTS_QK];
    float *k           = new float[TOTAL_ELEMENTS_QK];
    float *v           = new float[TOTAL_ELEMENTS_VO];
    float *o_ref       = new float[TOTAL_ELEMENTS_VO];
    float *o           = new float[TOTAL_ELEMENTS_VO];
    
    bf16 *q_bf        = new bf16[TOTAL_ELEMENTS_QK];
    bf16 *k_bf        = new bf16[TOTAL_ELEMENTS_QK];
    bf16 *v_bf        = new bf16[TOTAL_ELEMENTS_VO];
    bf16 *o_bf        = new bf16[TOTAL_ELEMENTS_VO];

    // debug dump
    int TOTAL_ELEMENTS_DEBUGO = B * H * N * CHUNK_SIZE;
    float *o_debug_ref = new float[TOTAL_ELEMENTS_DEBUGO];
    float *o_debug = new float[TOTAL_ELEMENTS_DEBUGO];
    bf16 *o_debug_bf  = new bf16[TOTAL_ELEMENTS_DEBUGO];
    std::cout << "TOTAL_ELEMENTS_DEBUGO " << TOTAL_ELEMENTS_DEBUGO << std::endl;


    if (argc > 1) {
        std::ifstream infile(argv[1]);
        std::cout << "Reading input file: " << argv[1] << std::endl;

        // 1. Read slopes
        for(int i = 0; i < ATTN_H; i++) {
            infile >> slopes[i];
            printf("slopes[%d] = %f\n", i, slopes[i]);
        }
        std::cout << "Finished loading " << ATTN_H << " slopes" << std::endl;

        // 2. Read Q
        for(int i = 0; i < TOTAL_ELEMENTS_QK; i++) infile >> q[i];
        std::cout << "Finished loading " << TOTAL_ELEMENTS_QK << " elements of Q" << std::endl;

        // 3. Read K
        for(int i = 0; i < TOTAL_ELEMENTS_QK; i++) infile >> k[i];
        std::cout << "Finished loading " << TOTAL_ELEMENTS_QK << " elements of K" << std::endl;

        // 4. Read V
        for(int i = 0; i < TOTAL_ELEMENTS_VO; i++) infile >> v[i];
        std::cout << "Finished loading " << TOTAL_ELEMENTS_VO << " elements of V" << std::endl;

        // 5. Read O reference
        for(int i = 0; i < TOTAL_ELEMENTS_VO; i++) infile >> o_ref[i];
        std::cout << "Finished loading " << TOTAL_ELEMENTS_VO << " elements of O_REF" << std::endl;

        // read debug dump
        for(int i = 0; i < TOTAL_ELEMENTS_DEBUGO; i++) {
            infile >> o_debug_ref[i];
            // if (i < 8) std::cout << "read o_debug_ref[" << i << "] " << o_debug_ref[i] << std::endl;
        }
        std::cout << "Finished loading " << TOTAL_ELEMENTS_DEBUGO << " elements of O_DEBUG_REF" << std::endl;
    }

    // Convert to bf16
    for(uint64_t i = 0; i < TOTAL_ELEMENTS_QK; i++) {
        q_bf[i] = __float2bfloat16(q[i]);
        k_bf[i] = __float2bfloat16(k[i]);
    }
    for(uint64_t i = 0; i < TOTAL_ELEMENTS_VO; i++) {
        v_bf[i] = __float2bfloat16(v[i]);
    }
    
    bf16 *d_q, *d_k, *d_v, *d_o;
    bf16 *d_debug_o; // used for debug dump only
    float *d_slopes;
    
    hipMalloc(&d_slopes,   H            * sizeof(float));
    hipMalloc(&d_q,        TOTAL_ELEMENTS_QK * sizeof(bf16));
    hipMalloc(&d_k,        TOTAL_ELEMENTS_QK * sizeof(bf16));
    hipMalloc(&d_v,        TOTAL_ELEMENTS_VO * sizeof(bf16));
    hipMalloc(&d_o,        TOTAL_ELEMENTS_VO * sizeof(bf16));

    hipMemcpy(d_slopes, slopes,   H            * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_q,      q_bf,     TOTAL_ELEMENTS_QK * sizeof(bf16),  hipMemcpyHostToDevice);
    hipMemcpy(d_k,      k_bf,     TOTAL_ELEMENTS_QK * sizeof(bf16),  hipMemcpyHostToDevice);
    hipMemcpy(d_v,      v_bf,     TOTAL_ELEMENTS_VO * sizeof(bf16),  hipMemcpyHostToDevice);
    
    // zero out d_o
    hipMemset(d_o, 0, TOTAL_ELEMENTS_VO * sizeof(bf16));

    // debug dump
    hipMalloc(&d_debug_o, TOTAL_ELEMENTS_DEBUGO * sizeof(bf16));
    hipMemset(d_debug_o, 0, TOTAL_ELEMENTS_DEBUGO * sizeof(bf16));

    hipDeviceSynchronize();
    HipCheckError();

    // Set up kernel configuration
    unsigned long mem_size = kittens::MAX_SHARED_MEMORY; 

    // Initialize kernel configuration
    lightning_attn2_globals g = lightning_attn2_init(
        d_q, d_k, d_v, d_o,
        // debug dump
        d_debug_o,
        d_slopes,
        B, H, N
    );

    hipFuncSetAttribute(
        (void*)qkv_kernel,
        hipFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );

    // Run kernel
    const int ITER = 1;
    hipDeviceSynchronize();
    HipCheckError();

    std::cout << "Starting kernel with " << B * H << " blocks and " << NUM_THREADS << " threads\n";
    float avg_us = 0;
    for(int i = 0; i < ITER; i++) {
        // zero out d_o
        hipMemset(d_o, 0, TOTAL_ELEMENTS_VO * sizeof(bf16));
        // debug dump
        hipMemset(d_debug_o, 0, TOTAL_ELEMENTS_DEBUGO * sizeof(bf16));
        hipDeviceSynchronize();
        HipCheckError();

        const auto start = std::chrono::high_resolution_clock::now();
        // lightning_attn2_kernel<<<g.grid(), g.block(), mem_size>>>(g, N);
        qkv_kernel<<<g.grid(), g.block(), mem_size>>>(g, N);
        hipDeviceSynchronize();
        const auto finish = std::chrono::high_resolution_clock::now();
        HipCheckError();
        avg_us += std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count();
    }
    avg_us /= ITER;
    std::cout << "Average execution time: " << avg_us << " us" << std::endl;

    // Copy results back and compare
    hipMemcpy(o_bf, d_o, TOTAL_ELEMENTS_VO * sizeof(bf16), hipMemcpyDeviceToHost);
    
    // Convert output to float
    for(int i = 0; i < TOTAL_ELEMENTS_VO; i++) {
        o[i] = __bfloat162float(o_bf[i]);
    }

    // debug dump
    hipMemcpy(o_debug_bf, d_debug_o, TOTAL_ELEMENTS_DEBUGO * sizeof(bf16), hipMemcpyDeviceToHost);
    for(int i = 0; i < TOTAL_ELEMENTS_DEBUGO; i++) {
        o_debug[i] = __bfloat162float(o_debug_bf[i]);
    }
    std::ofstream o_debug_ref_file("printouts/o_debug_ref.txt");
    std::ofstream o_debug_file("printouts/o_debug.txt");
    std::ofstream debug_diff_file("printouts/debug_sdiff.txt");
    float max_debug_diff = 0, total_debug_diff = 0, total_debug_abs = 0;
    for(int i = 0; i < TOTAL_ELEMENTS_DEBUGO; i++) {
        float debug_diff = o_debug[i] - o_debug_ref[i];
        if (i < 8) {
            std::cout << "o_debug[" << i << "] = " << o_debug[i]
              << " o_debug_ref[" << i << "] = " << o_debug_ref[i] << std::endl;
        }
        // else {
        //     std::cout << "o_debug[" << i << "] = " << o_debug[i]
        //       << " o_debug_ref[" << i << "] = " << o_debug_ref[i] << std::endl;
        // }
        
        o_debug_ref_file << o_debug_ref[i] << ' ';
        o_debug_file << o_debug[i] << ' ';
        debug_diff_file << debug_diff << ' ';
        
        if(i % 64 == 63) {
            o_debug_ref_file << std::endl;
            o_debug_file << std::endl;
            debug_diff_file << std::endl;
        }

        if(abs(debug_diff) > max_debug_diff || std::isnan(debug_diff)) {
            max_debug_diff = abs(debug_diff);
            if(std::isnan(debug_diff)) {
                printf("NAN detected idx=%d, o_debug = %f, o_debug_ref = %f, debug_diff = %f\n", i, o_debug[i], o_debug_ref[i], debug_diff);
                break;
            }
        }

        total_debug_abs += abs(o_debug_ref[i]);
        total_debug_diff += abs(debug_diff);
    }
    // Print error metrics
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout.precision(6);
    std::cout.width(12);
    std::cout << "total_debug_diff=" << total_debug_diff
              << "total_debug_abs=" << total_debug_abs << std::endl;
    std::cout << "O | avg_diff=" << (total_debug_diff/TOTAL_ELEMENTS_DEBUGO) 
              << ", avg_abs=" << (total_debug_abs/TOTAL_ELEMENTS_DEBUGO)
              << ", rel_diff=" << 100*(total_debug_diff/total_debug_abs) 
              << "%, max_debug_diff=" << max_debug_diff << std::endl;




    // Write results to files for analysis
    std::ofstream o_ref_file("printouts/o_ref.txt");
    std::ofstream o_file("printouts/o.txt");
    std::ofstream diff_file("printouts/diff.txt");

    float max_diff = 0, total_diff = 0, total_abs = 0;
    for(int i = 0; i < TOTAL_ELEMENTS_VO; i++) {
        float diff = o[i] - o_ref[i];
        if (i < 8) {
            std::cout << "o[" << i << "] = " << o[i]
              << " o_ref[" << i << "] = " << o_ref[i] << std::endl;
        }
        // else {
        //     std::cout << "o[" << i << "] = " << o[i]
        //       << " o_ref[" << i << "] = " << o_ref[i] << std::endl;
        // }
        
        o_ref_file << o_ref[i] << ' ';
        o_file << o[i] << ' ';
        diff_file << diff << ' ';
        
        if(i % 64 == 63) {
            o_ref_file << std::endl;
            o_file << std::endl;
            diff_file << std::endl;
        }

        if(abs(diff) > max_diff || std::isnan(diff)) {
            max_diff = abs(diff);
            if(std::isnan(diff)) {
                printf("NAN detected idx=%d, o = %f, o_ref = %f, diff = %f\n", i, o[i], o_ref[i], diff);
                break;
            }
        }

        total_abs += abs(o_ref[i]);
        total_diff += abs(diff);
    }

    // Print error metrics
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout.precision(6);
    std::cout.width(12);
    std::cout << "total_diff=" << total_diff
              << "total_abs=" << total_abs << std::endl;
    std::cout << "O | avg_diff=" << (total_diff/TOTAL_ELEMENTS_VO) 
              << ", avg_abs=" << (total_abs/TOTAL_ELEMENTS_VO)
              << ", rel_diff=" << 100*(total_diff/total_abs) 
              << "%, max_diff=" << max_diff << std::endl;

    // Cleanup
    hipFree(d_q);
    hipFree(d_k);
    hipFree(d_v);
    hipFree(d_o);
    hipFree(d_slopes);

    delete[] slopes;
    delete[] q;
    delete[] k;
    delete[] v;
    delete[] o;
    delete[] o_ref;
    delete[] q_bf;
    delete[] k_bf;
    delete[] v_bf;
    delete[] o_bf;

    // debug dump
    hipFree(d_debug_o);
    delete[] o_debug;
    delete[] o_debug_ref;
    delete[] o_debug_bf;

    return 0;
}