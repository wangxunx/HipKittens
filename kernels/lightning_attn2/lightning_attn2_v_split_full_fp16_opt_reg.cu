#include "kittens.cuh"
// #include "pyutils/pyutils.cuh"
#include <fstream>
#include <chrono>
#include <cmath>

// #define DEBUG

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

constexpr int CHUNK_SIZE = 64;

// debug
// #ifndef ATTN_B
// constexpr int ATTN_B = 1;//;//16; // batch size
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
// constexpr int ATTN_N = 1024;//128;//64;//1024; // sequence length
// #endif
// constexpr int CHUNK_SIZE = 64;


constexpr int V_CHUNK_SIZE = 32;

constexpr int CHUNK_SIZE_SPLIT = 2;
constexpr int KV_STATE_F_SPLIT = 2;

using namespace kittens;

using G = kittens::group<NUM_WARPS>;

#define BARRIER { \
    asm volatile("s_waitcnt vmcnt(0)"); \
    asm volatile("s_waitcnt lgkmcnt(0)"); \
    __builtin_amdgcn_s_barrier(); \
    __builtin_amdgcn_sched_barrier(0); \
    __syncthreads(); \
}

__device__ bool threadblock0() {
    return blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0;
}

__device__ bool thread0() {
    return threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0;
}

__device__ bool thread1() {
    return threadIdx.x == 1 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0;
}

__device__ bool thread2() {
    return threadIdx.x == 2 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0;
}

__device__ bool thread32() {
    return threadIdx.x == 32 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0;
}
__device__ bool thread33() {
    return threadIdx.x == 33 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0;
}

__device__ bool thread63() {
    return threadIdx.x == 63 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0;
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

// template<typename D>
// __device__ float sum_tile(D A2_tile){ 
//     float sum_val_A = 0;
//     #pragma unroll
//     for(int i = 0; i < A2_tile.height; i++){
//         #pragma unroll
//         for(int j = 0; j < A2_tile.width; j++){
//             // for(int k = 0)
//             #pragma unroll
//             for(int k = 0; k < A2_tile.packed_per_base_tile; k++){
//                 sum_val_A += A2_tile.tiles[i][j].data[k].x;
//                 // sum_val_A += A2_tile.tiles[i][j].data[k].y;
//             }
//         }
//     }

//     return sum_val_A;

// }

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

template<typename T, typename D>
__device__ T sum_rv(const D& vec){ 
    T sum_val_A = T(0);

    #pragma unroll
    for(int i = 0; i < D::outer_dim; i++){
        #pragma unroll
        for(int j = 0; j < D::inner_dim; j++){
            // #pragma unroll
            // for(int k = 0; k < D::packed_per_base_tile; k++){
            //     sum_val_A += static_cast<T>(vec.tiles[i][j].data[k].x);
            //     sum_val_A += static_cast<T>(vec.tiles[i][j].data[k].y);
            // }
            printf("%d, data[%d][%d].x %lf\n",  __LINE__,i,j, float(vec.data[i][j].x));
            printf("%d, data[%d][%d].y %lf\n",  __LINE__,i,j, float(vec.data[i][j].y));
            sum_val_A += static_cast<T>(vec.data[i][j].x);
            sum_val_A += static_cast<T>(vec.data[i][j].y);
        }
    }
    return sum_val_A;
}

template <int row=16, int col=16, int stride=col, typename T>
__device__ void print_smem(const T *ptr){
    if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) { 
        for(int i = 0; i < row; i ++) {
            if(i % 8 == 0 && i != 0) {
                printf("\n");
            }
            for(int j = 0; j < col; j++) {
                if(j % 8 == 0 && j != 0) {
                    printf("  ");
                }
                
                float now = float(ptr[i*stride+j]);
                printf("%7.4lf ",  now);
            }
            printf("\n");
        }
    }
}

#define DUMP_KV_STATE_SUM(MSG) {                                      \
    if (threadIdx.x == 0 && threadIdx.y == 0) {                       \
        float kv_state_sum = 0.0f;                                    \
        for (int i = 0; i < ATTN_F * V_CHUNK_SIZE; i++) {             \
            kv_state_sum += float(kv_state_smem.data[i]);             \
        }                                                             \
        printf("[%s] kv_state_sum = %f\n\n", MSG, kv_state_sum);      \
    }                                                                 \
}


template<int F, typename T=bf16, typename L=row_l, typename S=rt_32x16_s> using q_tile = rt<T, CHUNK_SIZE, F, L, S>;
template<int F, typename T=bf16, typename L=col_l, typename S=rt_16x32_s> using q_tile_transposed = rt<T, F, CHUNK_SIZE, L, S>;
template<int F, typename T=bf16, typename L=row_l, typename S=rt_32x16_s> using k_tile = rt<T, CHUNK_SIZE, F, L, S>;
// template<int D, typename T=bf16, typename L=row_l, typename S=rt_32x16_s> using v_tile = rt<T, CHUNK_SIZE, D, L, S>;
template<int D, typename T=bf16, typename L=row_l, typename S=rt_32x16_s> using v_tile = rt<T, CHUNK_SIZE, V_CHUNK_SIZE, L, S>;
template<int F, typename T=bf16, typename L=col_l, typename S=rt_16x32_s> using k_tile_transposed = rt<T, F, CHUNK_SIZE, L, S>;
template<int D, typename T=float, typename L=col_l, typename S=rt_16x32_4_s> using attn_tile = rt<T, CHUNK_SIZE, CHUNK_SIZE, L, S>;
// template<int D, typename T=bf16, typename L=row_l, typename S=rt_32x16_s> using o_tile = rt<T, CHUNK_SIZE, D, L, S>;
// template<int D, typename T=bf16, typename L=col_l, typename S=rt_16x32_s> using o_tile_transposed = rt<T, D, CHUNK_SIZE, L, S>;
template<int D, typename T=bf16, typename L=row_l, typename S=rt_32x16_s> using o_tile = rt<T, CHUNK_SIZE, V_CHUNK_SIZE, L, S>;
template<int D, typename T=bf16, typename L=col_l, typename S=rt_16x32_s> using o_tile_transposed = rt<T, V_CHUNK_SIZE, CHUNK_SIZE, L, S>;
template<int D, typename T=float, typename L=col_l, typename S=rt_16x32_s> using kv_state_tile = rt<T, ATTN_F, V_CHUNK_SIZE, L, S>;

using _gl_QKVO = gl<bf16, -1, -1, -1, -1>;

struct lightning_attn2_globals {
    _gl_QKVO Qg, Kg, K_split_g, Vg, Og;
    _gl_QKVO ODEBUGg;

    uintptr_t slopes;

    hipStream_t stream;
    dim3 block() {return dim3(NUM_THREADS);}
    // dim3 grid() {return dim3(ATTN_H, ATTN_B);}
    dim3 grid() {return dim3(ATTN_D / V_CHUNK_SIZE, ATTN_H, ATTN_B);} // parallel in V's ATTN_D dim
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

__device__ static inline bool check_register_allzero(auto &reg) {
    int height = reg.height;
    int width = reg.width;
    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            int packed_per_thread = reg.tiles[h][w].packed_per_thread;
            // int tile_data_length = sizeof(reg.tiles[h][w].data) / sizeof(reg.tiles[h][w].dtype);
            int tile_data_length = sizeof(reg.tiles[h][w].data) / sizeof(bf16_2);
            // printf("tile_data_length %d packed_per_thread %d sizeof(reg.tiles[h][w].data) %d\n", tile_data_length, packed_per_thread, sizeof(reg.tiles[h][w].data));
            if (tile_data_length != packed_per_thread) {
                printf("tile_data_length != packed_per_thread, error\n");
                return false;
            }
            for (int idx = 0; idx < packed_per_thread; idx++) {
                uint16_t val_bits_x = *reinterpret_cast<uint16_t*>(&reg.tiles[h][w].data[idx].x);
                uint16_t val_bits_y = *reinterpret_cast<uint16_t*>(&reg.tiles[h][w].data[idx].y);
                bool is_all_zero = ((val_bits_x == 0) && (val_bits_y == 0));
                if (is_all_zero == false)
                    return false;
            }
        }
    }
    return true;
}

__device__ static inline bool check_register_allone(auto &reg) {
    int height = reg.height;
    int width = reg.width;
    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            int packed_per_thread = reg.tiles[h][w].packed_per_thread;
            // int tile_data_length = sizeof(reg.tiles[h][w].data) / sizeof(reg.tiles[h][w].dtype);
            int tile_data_length = sizeof(reg.tiles[h][w].data) / sizeof(bf16_2);
            // printf("tile_data_length %d packed_per_thread %d sizeof(reg.tiles[h][w].data) %d\n", tile_data_length, packed_per_thread, sizeof(reg.tiles[h][w].data));
            if (tile_data_length != packed_per_thread) {
                printf("tile_data_length != packed_per_thread, error\n");
                return false;
            }
            for (int idx = 0; idx < packed_per_thread; idx++) {
                uint16_t val_bits_x = *reinterpret_cast<uint16_t*>(&reg.tiles[h][w].data[idx].x);
                uint16_t val_bits_y = *reinterpret_cast<uint16_t*>(&reg.tiles[h][w].data[idx].y);
                bool is_all_one = ((val_bits_x == 0x3F80) && (val_bits_y == 0x3F80));
                if (is_all_one == false)
                    return false;
            }
        }
    }
    return true;
}

__device__ static inline void dump_bits(bf16 * val) {
    float val_f = float(val[0]);
    uint16_t val_bits = *reinterpret_cast<uint16_t*>(val);

    // 快速打印核心信息
    printf("float value: %f → 十六进制: 0x%04X → 二进制：", val_f, val_bits);
    // 快速输出二进制字符串（16位）
    for (int i = 15; i >= 0; i--) {
        printf("%d", (val_bits >> i) & 1);
        if ((i % 8) == 7) printf(" "); // 每8位加空格，便于阅读
    }
    printf("\n");
    // bool is_all_zero = (val_bits == 0); // 所有bit位全0 → true
    // if (!is_all_zero) {
    //     printf("non-zero value found.\n");
    // }
}

__device__ static inline void dump_bits(float * val) {
    float val_f = float(val[0]);
    uint32_t val_bits = *reinterpret_cast<uint32_t*>(val);

    // 快速打印核心信息
    printf("float value: %f → 十六进制: 0x%08X → 二进制：", val_f, val_bits);
    // 快速输出二进制字符串（16位）
    for (int i = 31; i >= 0; i--) {
        printf("%d", (val_bits >> i) & 1);
        if ((i % 8) == 7) printf(" "); // 每8位加空格，便于阅读
    }
    printf("\n");
    // bool is_all_zero = (val_bits == 0); // 所有bit位全0 → true
    // if (!is_all_zero) {
    //     printf("non-zero value found.\n");
    // }
}

__device__ static inline void wg_arange(auto &vec) {
    #pragma unroll
    for (int i = 0; i < vec.length; i++) {
        // float val = static_cast<float>(i) + (warpid() * vec.length); 
        float val = static_cast<float>(i);
        vec.data[i] = val; 
    }
    // group<4>::sync(5 + warpgroupid());
}
// __device__ static inline void wg_arange(auto &vec) {
//     // #pragma unroll
//     // for (int i = 0; i < vec.length; i++) {
//     //     float val = static_cast<float>(i) + (warpid() * vec.length); 
//     //     vec.data[i] = val; 
//     // }
//     if (threadIdx.x < CHUNK_SIZE)
//         vec.data[threadIdx.x] = threadIdx.x;
// }

__device__ static inline void kv_state_init(auto &kv_state) {
    int rows = kv_state.rows;
    int cols = kv_state.cols;
    for (int i = 0; i < rows*cols; i++) {
        kv_state.data[i] = __float2bfloat16(1.0f);
    }
}

__device__ static inline float get_scale(int i, int j, float slope) {
    // float ret = __expf(-(i-j) * slope);
    // if (i < j) ret *= 0;

    // As attn_block is Q^TK, not QK^T, let's exchange i and j.
    float ret = __expf(-(j-i) * slope);
    if (j < i) ret *= 0;
        return ret;
}

template<ducks::rt::row_layout RT>
__device__ static inline void apply_mask(RT &dst, const RT &src, float slope) {

}

// __global__ __launch_bounds__(NUM_THREADS, 1)
__global__ __launch_bounds__(NUM_THREADS, 2)
void lightning_attn2_kernel(const lightning_attn2_globals globals, int N)
{
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    // smem
    st_bf<CHUNK_SIZE, ATTN_F, st_32x32_s> (&q_smem)[2] = al.allocate<st_bf<CHUNK_SIZE, ATTN_F, st_32x32_s>, 2>();                   // 64x128 x2 x2 = 32768 B = 32 KB
    st_bf<CHUNK_SIZE, ATTN_F, st_32x32_s> (&k_smem)[2] = al.allocate<st_bf<CHUNK_SIZE, ATTN_F, st_32x32_s>, 2>();                   // 64x128 x2 x2 = 32768 B = 32 KB
    st_bf<CHUNK_SIZE, V_CHUNK_SIZE, st_32x32_s> (&v_smem)[2] = al.allocate<st_bf<CHUNK_SIZE, V_CHUNK_SIZE, st_32x32_s>, 2>();       // 64x32 x2 x2 = 8192 B = 8 KB
    st_bf<ATTN_F, V_CHUNK_SIZE, st_32x32_s> (&kv_state_smem) = al.allocate<st_bf<ATTN_F, V_CHUNK_SIZE, st_32x32_s>>();              // 128x32 x2 = 8192B = 8KB

    constexpr int sizeof_shared = sizeof(q_smem) + sizeof(k_smem) + sizeof(v_smem) + sizeof(kv_state_smem);// + sizeof(kv_state_smem222);
    static_assert(sizeof_shared < 160000);

    row_vec<st_bf<ATTN_D, CHUNK_SIZE, st_32x32_s>> (&q_decay) = al.allocate<row_vec<st_bf<ATTN_D, CHUNK_SIZE, st_32x32_s>>>();      // 64x2 = 128 B
    row_vec<st_bf<ATTN_D, CHUNK_SIZE, st_32x32_s>> (&k_decay) = al.allocate<row_vec<st_bf<ATTN_D, CHUNK_SIZE, st_32x32_s>>>();      // 64x2 = 128 B
    // decay in register
    // row_vec<rt_fl<ATTN_D, CHUNK_SIZE, col_l, rt_32x32_s>> q_decay_rv;
    // col_vec<rt_fl<CHUNK_SIZE, ATTN_D, col_l, rt_32x32_s>> k_decay_rv;

    const int head_idx = blockIdx.y;
    const int batch_idx = blockIdx.z;
    const int v_block_idx = blockIdx.x;
    float slope = reinterpret_cast<float*>(globals.slopes)[head_idx];

    int blocks = N / CHUNK_SIZE;
    const int tic = 0, toc = 1;

    // Initialize all of the register tiles.
    q_tile<ATTN_F, bf16> q_reg;                         // [CHUNK_SIZE, ATTN_F], 64x128,                            64x128/64/2 = 64 VGPRs
    q_tile_transposed<ATTN_F, bf16> q_reg_transposed;   // [ATTN_F, CHUNK_SIZE], 128x64, rt_16x32_s, 8x2 subtiles,  64 VGPRs
    k_tile<ATTN_F, bf16> k_reg;                         // [CHUNK_SIZE, ATTN_F], 64x128,                            64x128/64/2 = 64 VGPRs
    k_tile_transposed<ATTN_F, bf16> k_reg_transposed;   // [ATTN_F, CHUNK_SIZE], 128x64,                            64 VGPRs
    
    v_tile<ATTN_D, bf16, col_l, rt_32x32_s> v_reg;                      // [CHUNK_SIZE, V_CHUNK_SIZE], 64x32,       64x32/64/2 = 16 VGPRs
    o_tile_transposed<ATTN_D, float, col_l, rt_32x32_s> o_intra;        // [V_CHUNK_SIZE, CHUNK_SIZE], 32x64,       32x64/64 = 32 VGPRs
    // o_tile_transposed<ATTN_D, float, col_l, rt_32x32_s> o_inter;        // [V_CHUNK_SIZE, CHUNK_SIZE], 32x64,       32x64/64 = 32 VGPRs
    attn_tile<ATTN_D, float, col_l, rt_32x32_s> attn_block[1];          // [CHUNK_SIZE, CHUNK_SIZE], 64x64,         64x64/64 = 64 VGPRs
    attn_tile<ATTN_D, bf16, col_l, rt_32x32_s> attn_block_bf16;         // [CHUNK_SIZE, CHUNK_SIZE], 64x64,         64x64/64/2 = 32 VGPRs

    // rt_bf<ATTN_F, V_CHUNK_SIZE, col_l, rt_16x32_s> local_kv_reg; // [ATTN_F, V_CHUNK_SIZE], 128x32, 8x1 subtiles
    rt_bf<ATTN_F/4, V_CHUNK_SIZE, col_l, rt_16x32_s> local_kv_reg; // [ATTN_F/4, V_CHUNK_SIZE], 32x32, 2x1 subtiles,32x32/64/2 = 8 VGPRs
    

    zero(o_intra);
    // zero(kv_state_smem);
    // zero(local_kv_reg);
    // ones(local_kv_reg);
    // store(kv_state_smem, local_kv_reg);
    if (warpid() == 0) {
        kv_state_init(kv_state_smem);
    }
    BARRIER;
    
#ifdef DEBUG
    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        // tile[0][0]
        // for (int i = 0; i < 4; i++) {
        //     if (threadIdx.x == 0 && threadIdx.y == 0){
        //         printf("local_kv_reg height %d width %d\n", local_kv_reg.height, local_kv_reg.width); // 8, 4
        //         printf("local_kv_reg.tiles[0][0].packed_per_thread %d\n", local_kv_reg.tiles[0][0].packed_per_thread); // 4
        //         float temp = __bfloat162float((local_kv_reg.tiles[0][0].data[i].x)); // HIP_vector_type<float, 2>
        //         printf("local_kv_reg.tiles[0][0].data[%d].x: %f\n", i, temp);
        //         dump_bits(&local_kv_reg.tiles[0][0].data[i].x);
        //         temp = __bfloat162float((local_kv_reg.tiles[0][0].data[i].y));
        //         printf("local_kv_reg.tiles[0][0].data[%d].y: %f\n", i, temp);
        //         dump_bits(&local_kv_reg.tiles[0][0].data[i].y);
        //     }
        // }
        // for (int i = 0; i < 8; i++) {
        //     if (threadIdx.x == 0 && threadIdx.y == 0){
        //         printf("kv_state_smem %f\n", float(kv_state_smem.data[i]));
        //         dump_bits(&kv_state_smem.data[i]);
        //     }
        // }
        DUMP_KV_STATE_SUM("after kv_state_smem inited");
    }
#endif

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

    int warpid = kittens::warpid();
    if (warpid == 0) {
        wg_arange(q_decay);
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory"); // 等待所有共享内存操作完成 (必加)
        __builtin_amdgcn_s_barrier();                      // block内全线程栅栏同步 (必加)
        
        row_vec<rt_fl<ATTN_D, CHUNK_SIZE, col_l, rt_32x32_s>> q_decay_rv;
        load(q_decay_rv, q_decay);
        mul(q_decay_rv, q_decay_rv, -1.0f * slope);
        exp(q_decay_rv, q_decay_rv); // OPT: exp2 -> exp
        store(q_decay, q_decay_rv);
        BARRIER;
#ifdef DEBUG
        // if (blockIdx.x == 0 && blockIdx.y == 0) {
        // if (blockIdx.x == 5 && blockIdx.y == 0) {
        //     for (int i = 0; i < 8; i++) {
        //         if (threadIdx.x == 0 && threadIdx.y == 0) {
        //             printf("q_decay[%d] %f\n", i, q_decay.data[i]);
        //         }
        //     }
        //     for (int i = CHUNK_SIZE - 8; i < CHUNK_SIZE; i++) {
        //         if (threadIdx.x == 0 && threadIdx.y == 0) {
        //             printf("q_decay[%d] %f\n", i, q_decay.data[i]);
        //         }
        //     }
        // }
        // // if (blockIdx.x == 0 && blockIdx.y == 0) {
        // if (blockIdx.x == 5 && blockIdx.y == 0) {
        //     if (threadIdx.x == 0 && threadIdx.y == 0) {
        //         printf("q_decay_rv outer_dim %d inner_dim %d elements_per_thread %d packing %d\n",
        //             q_decay_rv.outer_dim, q_decay_rv.inner_dim, q_decay_rv.elements_per_thread, q_decay_rv.packing); // 2, 1, 16, 1
        //         printf("slope %f\n", slope);
        //     }
        // }
        // if (blockIdx.x == 0 && blockIdx.y == 0) {
        //     printf("blockIdx.x %d, threadIdx %d, value: %f value: %f\n", blockIdx.x, threadIdx.x, q_decay_rv.data[0][0], q_decay_rv.data[1][0]);
        // }
        // if (blockIdx.x == 5 && blockIdx.y == 0) {
        //     printf("blockIdx.x %d, threadIdx %d, value: %f value: %f\n", blockIdx.x, threadIdx.x, q_decay_rv.data[0][0], q_decay_rv.data[1][0]);
        // }
        
#endif
    }
    BARRIER;

    if (warpid == 1) {
        wg_arange(k_decay);
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory"); // 等待所有共享内存操作完成 (必加)
        __builtin_amdgcn_s_barrier();                      // block内全线程栅栏同步 (必加)

        col_vec<rt_fl<CHUNK_SIZE, ATTN_D, col_l, rt_32x32_s>> k_decay_rv;
        load(k_decay_rv, k_decay); // 0,1,2,...,63
#ifdef DEBUG
        // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     for (int i = 0; i < 8; i++) {
        //         if (threadIdx.x == 64 && threadIdx.y == 0) {
        //             printf("k_decay[%d] %f\n", i, k_decay.data[i]);
        //         }
        //     }
        //     for (int i = CHUNK_SIZE - 8; i < CHUNK_SIZE; i++) {
        //         if (threadIdx.x == 64 && threadIdx.y == 0) {
        //             printf("k_decay[%d] %f\n", i, k_decay.data[i]);
        //         }
        //     }
        // }
        // print_smem<>();
#endif
        mul(k_decay_rv, k_decay_rv, -1.0f);         // 0,-1,-2,...,-63
        add(k_decay_rv, k_decay_rv, CHUNK_SIZE);    // 64,63,62,...,1
        mul(k_decay_rv, k_decay_rv, -1.0f * slope); // -64lambda,-63lambda,...,-lambda
        exp(k_decay_rv, k_decay_rv);
        store(k_decay, k_decay_rv);
#ifdef DEBUG
        // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     if (threadIdx.x == 64 && threadIdx.y == 0) {
        //         printf("k_decay_rv outer_dim %d inner_dim %d elements_per_thread %d packing %d\n",
        //             k_decay_rv.outer_dim, k_decay_rv.inner_dim, k_decay_rv.elements_per_thread, k_decay_rv.packing); // 2, 8, 16, 2
        //         printf("slope %f\n", slope);
        //     }
        // }
        
        // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        // if (thread(64)) {
        //     // packed type, bf16_2
        //     printf("blockIdx.x %d, threadIdx %d, k_decay_rv.data[0][0-7] value: (%f, %f) (%f, %f) (%f, %f) (%f, %f) (%f, %f) (%f, %f) (%f, %f) (%f, %f)\n",
        //         blockIdx.x, threadIdx.x,
        //         k_decay_rv.data[0][0].x, k_decay_rv.data[0][0].y, k_decay_rv.data[0][1].x, k_decay_rv.data[0][1].y, 
        //         k_decay_rv.data[0][2].x, k_decay_rv.data[0][2].y, k_decay_rv.data[0][3].x, k_decay_rv.data[0][3].y, 
        //         k_decay_rv.data[0][4].x, k_decay_rv.data[0][4].y, k_decay_rv.data[0][5].x, k_decay_rv.data[0][5].y, 
        //         k_decay_rv.data[0][6].x, k_decay_rv.data[0][6].y, k_decay_rv.data[0][7].x, k_decay_rv.data[0][7].y);
        //     printf("blockIdx.x %d, threadIdx %d, k_decay_rv.data[1][0-7] value: (%f, %f) (%f, %f) (%f, %f) (%f, %f) (%f, %f) (%f, %f) (%f, %f) (%f, %f)\n",
        //         blockIdx.x, threadIdx.x,
        //         k_decay_rv.data[1][0].x, k_decay_rv.data[1][0].y, k_decay_rv.data[1][1].x, k_decay_rv.data[1][1].y, 
        //         k_decay_rv.data[1][2].x, k_decay_rv.data[1][2].y, k_decay_rv.data[1][3].x, k_decay_rv.data[1][3].y, 
        //         k_decay_rv.data[1][4].x, k_decay_rv.data[1][4].y, k_decay_rv.data[1][5].x, k_decay_rv.data[1][5].y, 
        //         k_decay_rv.data[1][6].x, k_decay_rv.data[1][6].y, k_decay_rv.data[1][7].x, k_decay_rv.data[1][7].y);
        // }
        // if (blockIdx.x == 5 && blockIdx.y == 0) {
        //     // printf("blockIdx.x %d, threadIdx %d, k_decay_rv value: %f value: %f\n", blockIdx.x, threadIdx.x, k_decay_rv.data[0][0], k_decay_rv.data[1][0]);
        //     printf("blockIdx.x %d, threadIdx %d, k_decay_rv value: %f value: %f\n", blockIdx.x, threadIdx.x, k_decay_rv.data[0][0].x, k_decay_rv.data[0][0].y);
        // }
#endif
    }
    BARRIER;

// #ifdef DEBUG
//     if (blockIdx.x == 0 && blockIdx.y == 0) {
//         if (threadIdx.x == 0 && threadIdx.y == 0){
//             DUMP_KV_STATE_SUM("before global->smem load");
//         }
//     }
// #endif

    for (int block = 0; block < blocks; block++) {
#ifdef DEBUG
        if (thread0()) {
            printf("\n\n=======================>block: %d\n", block);
        }
#endif
        zero(o_intra);
        // zero(o_inter);
        // // Load Q, K, V tiles from global memory to shared memory
        G::load<1, false>(q_smem[tic], globals.Qg, {batch_idx, block, head_idx, 0}, swizzled_offsets_Q);
        G::load<1, false>(k_smem[tic], globals.Kg, {batch_idx, block, head_idx, 0}, swizzled_offsets_K);
        G::load<1, false>(v_smem[tic], globals.Vg, {batch_idx, block, head_idx, v_block_idx}, swizzled_offsets_V);
        BARRIER;
#ifdef DEBUG
        if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
            DUMP_KV_STATE_SUM("after v global->smem load");
        }
#endif

        // Below 3 loads work...
        // load<1, q_tile<ATTN_F, bf16>, _gl_QKVO>(q_reg, globals.Qg, {batch_idx, block, head_idx, 0});
        // load<1, k_tile<ATTN_F, bf16>, _gl_QKVO>(k_reg, globals.Kg, {batch_idx, block, head_idx, 0});
        // load<1, v_tile<ATTN_D, bf16, col_l, rt_16x32_4_s>, _gl_QKVO>(v_reg, globals.Vg, {batch_idx, block, head_idx, 0});        

        // smem to reg
        load(q_reg, q_smem[tic]);
        load(k_reg, k_smem[tic]);
        load(v_reg, v_smem[tic]);
        BARRIER;
#ifdef DEBUG
        if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
            DUMP_KV_STATE_SUM("after load qkv reg");
        }
#endif

        // calculation QK
        zero(attn_block[tic]);
        transpose(q_reg_transposed, q_reg);
        transpose(k_reg_transposed, k_reg);
        mma_AtB(attn_block[0], k_reg_transposed, q_reg_transposed, attn_block[0]);

        // __builtin_amdgcn_sched_barrier(0);
// #ifdef DEBUG
//         if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
//             DUMP_KV_STATE_SUM("after QK mma");
//         }
// #endif

        // apply diag decay
        // TODO
#ifdef DEBUG2
        if (thread0()) {
            D(attn_block[0].num_packed);                // 2
            D(attn_block[0].num_elements);              // 4096
            D(attn_block[0].elements_per_thread);       // 64
            D(attn_block[0].packed_per_thread);         // 32
            D(attn_block[0].packed_per_base_tile);      // 8, 2个float pack成1个float2，16个elements per thread
            D(attn_block[0].elements_per_base_tile);    // 16
        }
#endif
        const int lane_id = threadIdx.x % 64;
        for (int tr = 0; tr < attn_block[0].height; tr++){
            for (int tc = 0; tc < attn_block[0].width; tc++){
                for (int x = 0; x < attn_block[0].packed_per_base_tile; x++){
                    // TODO: do not hard so many numbers here
                    const float scale_x = get_scale(lane_id / 32 * 4 + x / 2 * 8 + x % 2 * 2 + tr * 32, lane_id % 32 + tc * 32, slope);
                    const float scale_y = get_scale(lane_id / 32 * 4 + x / 2 * 8 + x % 2 * 2 + 1 + tr * 32, lane_id % 32 + tc * 32, slope);
                    // if (threadblock0()) {
                    //     // printf("diag_decay (%d, %d) %f (%d, %d) %f\n", xi, xj, scale_x, yi, yj, scale_y);
                    //     printf("diag_decay (%d, %d) %.4e (%d, %d) %.4e\n", xi, xj, scale_x, yi, yj, scale_y);
                    // }
                    // if (thread(32)) {
                    //     printf("lane_id %d x %d (xi, xj):(%d, %d) (yi, yj):(%d, %d)\n", lane_id, x, xi, xj, yi, yj);
                    // }
                    // float x_temp = attn_block[0].tiles[tr][tc].data[x].x;
                    // float y_temp = attn_block[0].tiles[tr][tc].data[x].y;
                    attn_block[0].tiles[tr][tc].data[x].x *= scale_x;
                    attn_block[0].tiles[tr][tc].data[x].y *= scale_y;
                }
            }
        }

#ifdef DEBUG
        if (blockIdx.x == 0 && blockIdx.y == 0) {
            DUMP_KV_STATE_SUM("after diag_decay");
        }
#endif

        // TODO
        // attn_block_bf16 [CHUNK_SIZE, CHUNK_SIZE], 64x64
        copy(subtile_inplace<32>(attn_block_bf16, 0), subtile_inplace<32>(attn_block[0], 0));
        copy(subtile_inplace<32>(attn_block_bf16, 1), subtile_inplace<32>(attn_block[0], 1));
        BARRIER;
// #ifdef DEBUG
//         if (blockIdx.x == 0 && blockIdx.y == 0) {
//             DUMP_KV_STATE_SUM("after attn_block copy");
//         }
// #endif

        // calculate AV
        // v_reg [CHUNK_SIZE, V_CHUNK_SIZE], 64x32
        // o_intra [V_CHUNK_SIZE, CHUNK_SIZE], 32x64
        // v_reg * attn_block_bf16 -> o_intra, (64x32)^T * (64x64) = 32x64
        mma_AtB(o_intra, v_reg, attn_block_bf16, o_intra);

        // __builtin_amdgcn_s_setprio(0);
        // __builtin_amdgcn_sched_barrier(0);
        // __builtin_amdgcn_s_barrier();
        // __builtin_amdgcn_sched_barrier(0);
#ifdef DEBUG
        if (threadblock0()) {
            DUMP_KV_STATE_SUM("before load kv_state_smem");
        }
#endif

        /*
         * use subtile to reduce register
         */
        // O inter = q_reg * KV_state_sm

#ifdef DEBUG
        if (thread0()) {
            float o_inter_sum = sum_tile<float>(o_inter);
            D(o_inter_sum);
        }
#endif

        // row_vec<rt_fl<ATTN_D, CHUNK_SIZE, col_l, rt_32x32_s>> q_decay_rv;
        row_vec<rt_bf<ATTN_D, CHUNK_SIZE, col_l, rt_32x32_s>> q_decay_rv;
        load(local_kv_reg, subtile_inplace<ATTN_F/4, V_CHUNK_SIZE>(kv_state_smem, {0, 0}));
        BARRIER;
#ifdef DEBUG1
        if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
            DUMP_KV_STATE_SUM("after load kv_state_smem");
        }
#endif

        // mma_AtB(o_inter, local_kv_reg, q_reg_transposed, o_inter); // q_reg_transposed [ATTN_F, CHUNK_SIZE], 128x64, o_inter is (KV)^T*Q^T
        // merge o_inter and o_intra into one, to save register
        load(q_decay_rv, q_decay);
        BARRIER;
        mul_col(q_reg_transposed, q_reg_transposed, q_decay_rv);
        mma_AtB(o_intra, local_kv_reg, subtile_inplace<ATTN_F/4>(q_reg_transposed, 0), o_intra); // local_kv_reg [ATTN_F/4, V_CHUNK_SIZE], q_reg_transposed [ATTN_F, CHUNK_SIZE], 128x64, o_inter is (KV)^T*Q^T

        load(local_kv_reg, subtile_inplace<ATTN_F/4, V_CHUNK_SIZE>(kv_state_smem, {1, 0}));
        BARRIER;
        mma_AtB(o_intra, local_kv_reg, subtile_inplace<ATTN_F/4>(q_reg_transposed, 1), o_intra);

        load(local_kv_reg, subtile_inplace<ATTN_F/4, V_CHUNK_SIZE>(kv_state_smem, {2, 0}));
        BARRIER;
        mma_AtB(o_intra, local_kv_reg, subtile_inplace<ATTN_F/4>(q_reg_transposed, 2), o_intra);

        load(local_kv_reg, subtile_inplace<ATTN_F/4, V_CHUNK_SIZE>(kv_state_smem, {3, 0}));
        BARRIER;
        mma_AtB(o_intra, local_kv_reg, subtile_inplace<ATTN_F/4>(q_reg_transposed, 3), o_intra);
       
        load(q_decay_rv, q_decay);
        BARRIER;
#ifdef DEBUG
        if (thread0()) {
            float o_inter_sum = sum_tile<float>(o_inter);
            D(o_inter_sum);
        }
#endif
        // mul_col(o_intra, o_intra, q_decay_rv); // currently commented out for o_inter debug dump


        // update KV state
        // auto kv_subtile_0 = subtile_inplace<ATTN_F / 2, V_CHUNK_SIZE>(kv_state_smem, {0, 0});
        // auto kv_subtile_1 = subtile_inplace<ATTN_F / 2, V_CHUNK_SIZE>(kv_state_smem, {1, 0});
        
        // auto k_subtile_0 = subtile_inplace<CHUNK_SIZE, ATTN_F / 2>(k_smem[tic], {0, 0});
        // auto k_subtile_1 = subtile_inplace<CHUNK_SIZE, ATTN_F / 2>(k_smem[tic], {1, 0});
        
        // rt_fl<ATTN_F/2, ATTN_D> local_kv_0;
        // rt_fl<ATTN_F/2, ATTN_D> local_kv_1;
        // rt_fl<CHUNK_SIZE, ATTN_F/2> local_k_0;
        // rt_fl<CHUNK_SIZE, ATTN_F/2> local_k_1;

        rt_fl<ATTN_F, V_CHUNK_SIZE, col_l, rt_32x32_s> local_kv;        // 128x32，128x32/64 = 32 VGPRs
        // rt_bf<CHUNK_SIZE, ATTN_F, col_l, rt_32x32_s> local_k;           // 64x128,      64x128/64/2 = 64 VGPRs
        // optimize vgprs
        rt_bf<CHUNK_SIZE/CHUNK_SIZE_SPLIT, ATTN_F, col_l, rt_32x32_s> local_k;           // 32x128,      32x128/64/2 = 32 VGPRs

        load(local_k, subtile_inplace<CHUNK_SIZE/CHUNK_SIZE_SPLIT, ATTN_F>(k_smem[0], {0, 0}));
#ifdef DEBUG
        if (thread0()) {
            float local_k_sum = sum_tile<float>(local_k);
            D(local_k_sum);
        }
#endif
        col_vec<rt_bf<CHUNK_SIZE/CHUNK_SIZE_SPLIT, ATTN_D, col_l, rt_32x32_s>> k_decay_rv; // half
        load(k_decay_rv, subvec_inplace<CHUNK_SIZE/2>(k_decay, 0)); // 1st half
#ifdef DEBUG
        BARRIER;
        if (thread0()) {
            printf("1st half k_decay_rv outer_dim %d inner_dim %d elements_per_thread %d packing %d\n",
                    k_decay_rv.outer_dim, k_decay_rv.inner_dim, k_decay_rv.elements_per_thread, k_decay_rv.packing); // 1, 8, 16, 2
            float k_decay_rv_sum = sum_rv<float>(k_decay_rv);
            D(k_decay_rv_sum);
        }
#endif
        mul_row(local_k, local_k, k_decay_rv);
        // copy(local_k_bf16, local_k);
#ifdef DEBUG
        if (thread0()) {
            float local_k_sum = sum_tile<float>(local_k);
            D(local_k_sum);
        }
#endif
        float block_decay = __expf(-slope * static_cast<float>(CHUNK_SIZE));

        load(local_kv, kv_state_smem);
        mul(local_kv, local_kv, block_decay);
        // D: 128x128 float
        // A: k_reg_transposed [ATTN_F, CHUNK_SIZE], 128x64, bf16
        // A: k_reg [CHUNK_SIZE, ATTN_F], 64x128, bf16
        // A: local_k_bf16, [CHUNK_SIZE, ATTN_F], 64x128, bf16
        // B: [CHUNK_SIZE, ATTN_D], 64x128
        mma_AtB(local_kv, local_k, subtile_inplace<CHUNK_SIZE/CHUNK_SIZE_SPLIT>(v_reg, 0), local_kv);
        load(local_k, subtile_inplace<CHUNK_SIZE/CHUNK_SIZE_SPLIT, ATTN_F>(k_smem[0], {1, 0}));
        load(k_decay_rv, subvec_inplace<CHUNK_SIZE/2>(k_decay, 1)); // 2nd half
#ifdef DEBUG
        BARRIER;
        if (thread0()) {
            printf("2nd half k_decay_rv outer_dim %d inner_dim %d elements_per_thread %d packing %d\n",
                    k_decay_rv.outer_dim, k_decay_rv.inner_dim, k_decay_rv.elements_per_thread, k_decay_rv.packing); // 1, 8, 16, 2
            float k_decay_rv_sum = sum_rv<float>(k_decay_rv);
            D(k_decay_rv_sum);
        }
#endif
        mul_row(local_k, local_k, k_decay_rv);
        mma_AtB(local_kv, local_k, subtile_inplace<CHUNK_SIZE/CHUNK_SIZE_SPLIT>(v_reg, 1), local_kv);
        BARRIER;


        
        // store updated kv state
#ifdef DEBUG
        // PT(local_kv);
        if (thread(0)) {
            float local_kv_sum = sum_tile<float>(local_kv);
            D(local_kv_sum);
        }
        BARRIER;
#endif
        // TODO
        store(kv_state_smem, local_kv);
        BARRIER;

#ifdef DEBUG
        if (threadblock0()) {
            // print_smem<ATTN_F, V_CHUNK_SIZE>(kv_state_smem.data);
            DUMP_KV_STATE_SUM("after store updated local_kv to kv_state_smem");
        }
#endif

       // o_intra + o_inter
       // o_intra [V_CHUNK_SIZE, CHUNK_SIZE] 32x64, o_inter [V_CHUNK_SIZE, CHUNK_SIZE]
    //    add(o_intra, o_inter, o_intra); // commented out for o_intra w/o mask debug

        
        o_tile<ATTN_D, float, row_l, rt_32x32_s> o_reg_transposed; // [CHUNK_SIZE, V_CHUNK_SIZE]
        transpose(o_reg_transposed, o_intra);

        store<1>(globals.Og, o_reg_transposed, {batch_idx, block, head_idx, v_block_idx});

        // // debug dump o_inter
        // o_tile<ATTN_D, float, row_l, rt_32x32_s> o_inter_transposed; // [CHUNK_SIZE, V_CHUNK_SIZE]
        // transpose(o_inter_transposed, o_inter);
        // store<1>(globals.ODEBUGg, o_inter_transposed, {batch_idx, block, head_idx, v_block_idx}); // dump o_inter
        // // store<1>(globals.ODEBUGg, o_reg_transposed, {batch_idx, block, head_idx, v_block_idx}); // dump o_intra
    }
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
    _gl_QKVO            odebug_arg{d_debug_o, B, N, H, ATTN_D};

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

// Calculate FLOPs for Lightning Attention-2
// For matmul A(m,k) @ B(k,n), FLOPs = 2*m*k*n (multiply-add counts as 2 ops)
double calculate_lightning_attn2_flops(int B, int H, int N, int D, int F) {
    // const int CHUNK_SIZE = 64;
    const int num_blocks = N / CHUNK_SIZE;
    
    double flops_per_block = 0;
    
    // 1. QK^T: (CHUNK_SIZE, D) @ (D, CHUNK_SIZE) = (CHUNK_SIZE, CHUNK_SIZE)
    //    FLOPs = 2 * CHUNK_SIZE * D * CHUNK_SIZE
    flops_per_block += 2.0 * CHUNK_SIZE * D * CHUNK_SIZE;

    // 2. diag decay
    flops_per_block += CHUNK_SIZE * CHUNK_SIZE;
    
    // 3. Attention @ V: (CHUNK_SIZE, CHUNK_SIZE) @ (CHUNK_SIZE, F) = (CHUNK_SIZE, F)
    //    FLOPs = 2 * CHUNK_SIZE * CHUNK_SIZE * F
    flops_per_block += 2.0 * CHUNK_SIZE * CHUNK_SIZE * F;

    
    // 4. Q @ KV_state: (CHUNK_SIZE, D) @ (D, F) = (CHUNK_SIZE, F)
    //    FLOPs = 2 * CHUNK_SIZE * D * F
    flops_per_block += 2.0 * CHUNK_SIZE * D * F;

    // 5. Q decay
    flops_per_block += CHUNK_SIZE * F;
    
    // 6. K^T @ V (update KV state): (D, CHUNK_SIZE) @ (CHUNK_SIZE, F) = (D, F)
    //    FLOPs = 2 * D * CHUNK_SIZE * F
    flops_per_block += 2.0 * D * CHUNK_SIZE * F;

    // 7.K decay
    flops_per_block += CHUNK_SIZE * D;

    // Total FLOPs = flops_per_block * num_blocks * num_heads * batch_size
    double total_flops = flops_per_block * num_blocks * H * B;
    
    return total_flops;
}


int main(int argc, char **argv) {
    constexpr int B = 16;
    constexpr int D = 128;
    constexpr int H = 8;
    constexpr int F = 128;
    constexpr int N = 1024;

    // constexpr int B = 1;
    // constexpr int D = 128;
    // constexpr int H = 1;
    // constexpr int F = 128;
    // // constexpr int N = 64;
    // // constexpr int N = 128;
    // constexpr int N = 1024;

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
    int TOTAL_ELEMENTS_DEBUGO = B * H * N * D;
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
        for(int i = 0; i < TOTAL_ELEMENTS_DEBUGO; i++) infile >> o_debug_ref[i];
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
        (void*)lightning_attn2_kernel,
        hipFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );

    // Run kernel
    constexpr int WARMUP_ITERS = 5;
    constexpr int TIMING_ITERS = 10;
    hipDeviceSynchronize();
    HipCheckError();

    std::cout << "Starting kernel with " << B * H << " blocks and " << NUM_THREADS << " threads\n";

    // Warmup iterations
    std::cout << "Running " << WARMUP_ITERS << " warmup iterations..." << std::endl;
    for(int i = 0; i < WARMUP_ITERS; i++) {
        // zero out d_o
        hipMemset(d_o, 0, TOTAL_ELEMENTS_VO * sizeof(bf16));
        // debug dump
        hipMemset(d_debug_o, 0, TOTAL_ELEMENTS_DEBUGO * sizeof(bf16));
        hipDeviceSynchronize();
        
        lightning_attn2_kernel<<<g.grid(), g.block(), mem_size>>>(g, N);
        hipDeviceSynchronize();
        HipCheckError();
    }
    std::cout << "Warmup complete." << std::endl;

    // Timing iterations
    std::cout << "Running " << TIMING_ITERS << " timing iterations..." << std::endl;
    float total_us = 0;
    float min_us = std::numeric_limits<float>::max();
    float max_us = 0;
    std::vector<float> latencies;
    latencies.reserve(TIMING_ITERS);
    for(int i = 0; i < TIMING_ITERS; i++) {
        // zero out d_o
        hipMemset(d_o, 0, TOTAL_ELEMENTS_VO * sizeof(bf16));
        // debug dump
        hipMemset(d_debug_o, 0, TOTAL_ELEMENTS_DEBUGO * sizeof(bf16));
        hipDeviceSynchronize();
        HipCheckError();

        const auto start = std::chrono::high_resolution_clock::now();
        // lightning_attn2_kernel<<<dim3(H,B), NUM_THREADS, mem_size>>>(g, N);
        lightning_attn2_kernel<<<g.grid(), g.block(), mem_size>>>(g, N);
        hipDeviceSynchronize();
        const auto finish = std::chrono::high_resolution_clock::now();
        HipCheckError();
        float iter_us = std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count();
        latencies.push_back(iter_us);
        total_us += iter_us;
        min_us = std::min(min_us, iter_us);
        max_us = std::max(max_us, iter_us);
        
        std::cout << "  Iteration " << i << ": " << iter_us << " us" << std::endl;
    }
    float avg_us = total_us / TIMING_ITERS;
    // Calculate standard deviation
    float variance = 0;
    for(float latency : latencies) {
        variance += (latency - avg_us) * (latency - avg_us);
    }
    variance /= TIMING_ITERS;
    float stddev_us = std::sqrt(variance);
    // Print statistics
    std::cout << "\n=== Kernel Performance Statistics ===" << std::endl;
    std::cout << "Average latency: " << avg_us << " us" << std::endl;
    std::cout << "Minimum latency: " << min_us << " us" << std::endl;
    std::cout << "Maximum latency: " << max_us << " us" << std::endl;
    std::cout << "Std deviation:   " << stddev_us << " us" << std::endl;
    std::cout << "Throughput:      " << (1000000.0 / avg_us) << " iterations/s" << std::endl;
    std::cout << "====================================" << std::endl;
    // Calculate and print FLOPs metrics
    double total_flops = calculate_lightning_attn2_flops(B, H, N, ATTN_D, ATTN_F);
    double tflops = total_flops / 1e12;
    double avg_tflops_per_sec = tflops / (avg_us / 1e6);
    double peak_tflops_per_sec = tflops / (min_us / 1e6);
    
    std::cout << "\n=== FLOPs Analysis ===" << std::endl;
    std::cout << "Total FLOPs:     " << total_flops / 1e9 << " GFLOPs" << std::endl;
    std::cout << "                 " << tflops << " TFLOPs" << std::endl;
    std::cout << "Avg Performance: " << avg_tflops_per_sec << " TFLOPs/s" << std::endl;
    std::cout << "Peak Performance:" << peak_tflops_per_sec << " TFLOPs/s" << std::endl;
    std::cout << "====================================" << std::endl;

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