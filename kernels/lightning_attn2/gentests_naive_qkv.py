import torch
import torch.nn.functional as F
from tqdm import trange
import sys
import numpy as np

# from baselines.lightning_attn2 import lightning_attn2
from baselines.lightning_attn2_dump import lightning_attn2
from baselines.lightning_attn2_naive import lightning_attn2_naive

D_QK = 128
D_VO = 128
CHUNK_SIZE = 64


def generate_inputs(B, H, N):
    q = torch.randn((B, H, N, D_QK), dtype=torch.bfloat16, device='cuda') / (D_QK ** 0.5)
    k = torch.randn((B, H, N, D_QK), dtype=torch.bfloat16, device='cuda') / (D_QK ** 0.5)
    v = torch.randn((B, H, N, D_VO), dtype=torch.bfloat16, device='cuda')

    # q = torch.ones((B, H, N, D_QK), dtype=torch.bfloat16, device='cuda')
    # k = torch.ones((B, H, N, D_QK), dtype=torch.bfloat16, device='cuda')
    # k[:,:,0:N//2,:] = k[:,:,0:N//2,:] / 2
    # k[:,:,N//2:N,:] = k[:,:,N//2:N,:] / 4
    # v = torch.ones((B, H, N, D_VO), dtype=torch.bfloat16, device='cuda')
    # v[:,:,:,0:D_VO//2] = v[:,:,:,0:D_VO//2] / 4
    # v[:,:,:,D_VO//2:D_VO] = v[:,:,:,D_VO//2:D_VO] / 2

    # q = torch.ones((B, H, N, D_QK), dtype=torch.bfloat16, device='cuda')
    # k = torch.ones((B, H, N, D_QK), dtype=torch.bfloat16, device='cuda') * 2
    # v = torch.ones((B, H, N, D_VO), dtype=torch.bfloat16, device='cuda') * 3
    
    s = torch.rand((H,), dtype=torch.float32, device='cuda')  # s stays float32
    return q, k, v, s

def get_mask(n, slope=1):
    mask = torch.triu(torch.zeros(n, n).float().fill_(float("-inf")), 1)
    for i in range(n):
        x = torch.arange(i + 1)
        y = slope * x
        mask[i, :i + 1] = -torch.flip(y, [0])
    return torch.exp(mask)

def get_full_mask(n, slopes):
    arr = []
    for slope in slopes:
        arr.append(get_mask(n, slope.item()))
    mask = torch.stack(arr, dim=0)
    return mask

def linear_attn(q, k, v, s):
    b, h, n, d = q.shape
    mask = get_full_mask(n, s).to(q.device).to(torch.float32)
    qk = torch.matmul(q, k.transpose(2, 3))
    qk = (qk.to(torch.float32) * mask).to(q.dtype)
    o = torch.matmul(qk, v)
    return o

def linear_attn_naive_qkv(q, k, v):
    qk = torch.matmul(q, k.transpose(2, 3))
    o = torch.matmul(qk, v)
    return o

def linear_attn_naive_qkv_lightning_version(q, k, v):
    b, h, n, d = q.shape
    num_chunks = (n + CHUNK_SIZE - 1) // CHUNK_SIZE
    o = torch.empty_like(q)

    for i in range(num_chunks):
        start = i * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, n)
        q_chunk = q[:, :, start:end, :]
        k_chunk = k[:, :, start:end, :]
        v_chunk = v[:, :, start:end, :]
    
        qk = torch.matmul(q_chunk, k_chunk.transpose(2, 3))
        o_chunk = torch.matmul(qk, v_chunk)
        o[:, :, start:end, :] = o_chunk
    return o

def save_test_case(q, k, v, s, o, n, o_debug=None):
    filename = f'naive_qkv_randn_{n}.txt'
    print(f"slopes: {s}")
    import pdb
    pdb.set_trace()
    with open(filename, 'w') as f:    
        sf = s.to(torch.float32).flatten().cpu().numpy().tolist()
        qf = q.to(torch.float32).flatten().cpu().numpy().tolist()
        kf = k.to(torch.float32).flatten().cpu().numpy().tolist()
        vf = v.to(torch.float32).flatten().cpu().numpy().tolist()
        of = o.to(torch.float32).flatten().cpu().numpy().tolist()

        for i in trange(len(sf)):
            f.write(repr(sf[i]))
            f.write(' ')

        for i in trange(len(qf)):
            f.write(repr(qf[i]))
            f.write(' ')
            
        for i in trange(len(kf)):
            f.write(repr(kf[i]))
            f.write(' ')

        for i in trange(len(vf)):
            f.write(repr(vf[i]))
            f.write(' ')

        for i in trange(len(of)):
            f.write(repr(of[i]))
            f.write(' ')
        
        if o_debug is not None:
            o_debug_f = o_debug.to(torch.float32).flatten().cpu().numpy().tolist()
            for i in trange(len(o_debug_f)):
                f.write(repr(o_debug_f[i]))
                f.write(' ')


def main():
    torch.manual_seed(0)
    # torch.manual_seed(42)
    
    # B, H = 16, 8
    # sequence_lengths = [1024]

    B, H = 1, 1
    sequence_lengths = [64]
    # sequence_lengths = [128]
    # sequence_lengths = [1024]
    
    for N in sequence_lengths:
        print(f"\nGenerating test case for sequence length {N}")
        q, k, v, s = generate_inputs(B, H, N)

        # pytorch_out = linear_attn(q, k, v, s)
        # pytorch_out = linear_attn_naive_qkv(q, k, v)
        pytorch_out = linear_attn_naive_qkv_lightning_version(q, k, v)
        import pdb
        pdb.set_trace()
        triton_out, triton_debug_out, kv0_debug, kv1_debug = lightning_attn2(q, k, v, s)
        triton_out_naive, triton_debug_out_naive, kv0_naive, kv1_naive = lightning_attn2_naive(q, k, v, s)
        print(f"kv0_naive: {kv0_naive}")
        print(f"kv1_naive: {kv1_naive}")
        print(f"kv0_debug: {kv0_debug}")
        print(f"kv1_debug: {kv1_debug}")
        np.save("kv0_naive.npy", kv0_naive.cpu().numpy())
        np.save("kv1_naive.npy", kv1_naive.cpu().numpy())
        # np.save("triton_out_naive_block32.npy", triton_out_naive.to(torch.float32).cpu().numpy())
        pdb.set_trace()
        assert torch.allclose(triton_out, triton_out_naive, atol=1e-3, rtol=1e-3)
        assert torch.allclose(triton_debug_out, triton_debug_out_naive, atol=1e-3, rtol=1e-3)
        assert torch.allclose(kv0_debug, kv0_naive, atol=1e-5, rtol=1e-5)
        assert torch.allclose(kv1_debug, kv1_naive, atol=1e-5, rtol=1e-5)
        
        avg_mag_pytorch = torch.mean(torch.abs(pytorch_out)).item()
        # avg_mag_triton = torch.mean(torch.abs(triton_out)).item()
        # max_diff = torch.max(torch.abs(pytorch_out - triton_out)).item()
        # avg_diff = torch.mean(torch.abs(pytorch_out - triton_out)).item()
        # assert torch.allclose(pytorch_out, triton_out, atol=1e-3, rtol=1e-3)
        
        print(f"PyTorch output magnitude: {avg_mag_pytorch}")
        # print(f"Triton  output magnitude: {avg_mag_triton}")
        # print(f"Max     difference between PyTorch and Triton: {max_diff}")
        # print(f"Average difference between PyTorch and Triton: {avg_diff}")
        
        # save_test_case(q, k, v, s, triton_out, N)
        # save_test_case(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), s, pytorch_out.transpose(1, 2), N) # for amd layout
        # save_test_case(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), s, triton_out.transpose(1, 2), N) # for amd layout
        # debug dump
        save_test_case(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), s, triton_out.transpose(1, 2), N, triton_debug_out.transpose(1, 2)) # for amd layout
        print(f"Generated random test case for N={N}")

if __name__ == "__main__":
    main()