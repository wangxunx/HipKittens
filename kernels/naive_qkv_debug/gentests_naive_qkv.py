import torch
import torch.nn.functional as F
from tqdm import trange
import sys
# from baselines.lightning_attn2 import lightning_attn2

import numpy as np

D_QK = 128
D_VO = 128
CHUNK_SIZE = 64

def generate_inputs(B, H, N):
    q = torch.randn((B, H, N, D_QK), dtype=torch.bfloat16, device='cuda') / (D_QK ** 0.5)
    # q = q * 0.0 + 1
    k = torch.randn((B, H, N, D_QK), dtype=torch.bfloat16, device='cuda') / (D_QK ** 0.5)
    v = torch.randn((B, H, N, D_VO), dtype=torch.bfloat16, device='cuda')
    # v = v * 0.0 + 1
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
    import pdb; pdb.set_trace()
    # return o
    return qk

def linear_attn_naive_qk_lightning_version(q, k, v):
    b, h, n, d = q.shape
    num_chunks = (n + CHUNK_SIZE - 1) // CHUNK_SIZE
    o_qk = torch.zeros(b, h, n, CHUNK_SIZE).to(q.device).to(torch.bfloat16)

    for i in range(num_chunks):
        start = i * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, n)
        q_chunk = q[:, :, start:end, :]
        k_chunk = k[:, :, start:end, :]
        v_chunk = v[:, :, start:end, :]
    
        qk = torch.matmul(q_chunk, k_chunk.transpose(2, 3))
        import pdb; pdb.set_trace()
        o_chunk = torch.matmul(qk, v_chunk)
        o_qk[:, :, start:end, :] = qk
    return o_qk

def linear_attn_naive_qkv_lightning_version(q, k, v):
    b, h, n, d = q.shape
    num_chunks = (n + CHUNK_SIZE - 1) // CHUNK_SIZE
    # o_qk = torch.zeros(b, h, n, CHUNK_SIZE).to(q.device).to(torch.bfloat16)
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

def save_test_case(q, k, v, s, o, n):
    filename = f'naive_qkv_randn_{n}.txt'
    print(f"slopes: {s}")
    import pdb
    pdb.set_trace()

    # np.save("naive_qkv_input_q.npy", q.to(torch.float32).cpu().numpy())
    # np.save("naive_qkv_input_k.npy", k.to(torch.float32).cpu().numpy())
    # np.save("naive_qkv_input_v.npy", v.to(torch.float32).cpu().numpy())
    # np.save("naive_qkv_output_o.npy", o.to(torch.float32).cpu().numpy())

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

def main():
    torch.manual_seed(0)
    
    B, H = 16, 8
    sequence_lengths = [1024]

    # B, H = 1, 1
    # sequence_lengths = [64]
    # sequence_lengths = [128]
    # sequence_lengths = [1024]
    
    
    for N in sequence_lengths:
        print(f"\nGenerating test case for sequence length {N}")
        q, k, v, s = generate_inputs(B, H, N)

        # pytorch_out = linear_attn(q, k, v, s)
        # pytorch_out = linear_attn_naive_qkv(q, k, v)
        # pytorch_out = linear_attn_naive_qk_lightning_version(q, k, v)
        pytorch_out = linear_attn_naive_qkv_lightning_version(q, k, v)
        import pdb
        pdb.set_trace()
        # triton_out = lightning_attn2(q, k, v, s)
        
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
        save_test_case(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), s, pytorch_out.transpose(1, 2), N) # for amd layout
        print(f"Generated random test case for N={N}")

if __name__ == "__main__":
    main()