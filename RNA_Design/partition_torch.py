import os
import sys
import time
import argparse
import json
from collections import defaultdict

from partition_stoc import all_inside, rand_arr, pairs_index, nucs, inside_forward_stoc,inside_forward, inside_forward_pt, sharpturn, set_sharpturn, pair2score, unpair_score, kT

import numpy as np
import torch

# kT = 61.63207755
# kT = 1


def projection_simplex_torch_batch(x, z=1, device='cpu'):
    x_sorted, _ = torch.sort(x, dim=1, descending=True)
    cumsum = torch.cumsum(x_sorted, axis=1)
    denom = torch.arange(x.shape[1], device=device) + 1
    theta = (cumsum - z)/denom
    mask = x_sorted > theta 
    csum_mask = torch.cumsum(mask, axis=1)
    index = csum_mask[:, -1] - 1
    # print(f"index.shape: {index.shape}, theta.shape: {theta.shape}")
    x_proj = torch.maximum(x.transpose(0, 1) - theta[np.arange(len(x)), index], torch.tensor(0))
    return x_proj.transpose(0, 1)


def inside_forward_stoc_log_torch(s, device='cpu'): # s: n*4 matrix, row sum-to-one; A C G U
    assert len(s) > 1, "the length of rna should be at least 2!"
    n = len(s)
    counts = defaultdict(lambda: defaultdict(lambda: torch.tensor(-torch.inf, device=device))) # number of structures
    for k in range(n):
        counts[k][k]= torch.tensor(0., device=device)
        counts[k][k-1] = torch.tensor(0., device=device)
    for j in range(1, n):
        for i in range(0, j):
            counts[i][j] = torch.logaddexp(counts[i][j], counts[i][j-1]) # x.
            if j-i>sharpturn:
                prob = torch.tensor(0., device=device)
                for il, jr in pairs_index: 
                    prob += s[i][il]*s[j][jr]
                if prob > 0:
                    prob = torch.log(prob)
                    counts_right = counts[i+1][j-1] 
                    for t in range(0, i):
                        counts_left = counts[t][i-1] 
                        counts[t][j] = torch.logaddexp(counts[t][j], counts_left+counts_right+prob) # x(x)
                    counts[i][j] = torch.logaddexp(counts[i][j], counts_right+prob) # (x)
    return counts


def inside_forward_pt_stoc_log_torch(s, device='cpu'): # s: n*4 matrix, row sum-to-one; A C G U
    assert len(s) > 1, "the length of rna should be at least 2!"
    # print(f'sharpturn: {sharpturn}')
    n = len(s)
    counts = defaultdict(lambda: defaultdict(lambda: torch.tensor(-torch.inf, device=device))) # number of structures
    for k in range(n):
        counts[k][k]= torch.tensor(-unpair_score/kT, device=device)
        counts[k][k-1] = torch.tensor(0., device=device)
    for j in range(1, n):
        for i in range(0, j):
            counts[i][j] = torch.logaddexp(counts[i][j], counts[i][j-1]+(-unpair_score/kT)) # x.
            if j-i>sharpturn:
                score_ij = torch.tensor(-torch.inf, device=device)
                prob_sum = 0
                for il, jr in pairs_index: 
                    prob = s[i][il]*s[j][jr]
                    prob_sum += prob
                    if prob > 0:
                        pair_ij = nucs[il]+nucs[jr]
                        score_ij = torch.logaddexp(score_ij, torch.log(prob)+(-pair2score[pair_ij]/kT))
                if prob_sum > 0:
                    counts_right = counts[i+1][j-1] 
                    for t in range(0, i):
                        counts_left = counts[t][i-1] 
                        counts[t][j] = torch.logaddexp(counts[t][j], counts_left+counts_right+score_ij) # x(x)
                    counts[i][j] = torch.logaddexp(counts[i][j], counts_right+score_ij) # (x)
    return counts


def max_count(l, lr):
    ts = torch.tensor(rand_arr(l), requires_grad=True)
    log = []
    for epoch in range(100):
        ts.grad = None
        inside = inside_forward_stoc(ts)
        count = -inside[0][l-1]
        print('count: ', -count.item())
        count.backward()
        # print(ts.grad)
        ts_next = ts - ts.grad*lr
        ts = projection_simplex_torch_batch(ts_next.detach())
        ts.requires_grad_(True)
        log.append(count.item())
    print('optimal solution:')
    for i in range(l):
        print(f"A: {ts[i][0]:.4f}, C: {ts[i][1]:.4f}, G: {ts[i][2]:.4f}, U: {ts[i][3]:.4f}")
    print('optimal sequence:')
    seq = "".join([nucs[index] for index in torch.argmax(ts, axis=1)])
    print(seq)
    print('count:', inside_forward(seq)[0][l-1])
    return ts, log


def min_log_count(l, lr, num_step, init=None, log_last=None, device='cpu'):
    ts = torch.tensor(rand_arr(l), requires_grad=True, device=device) if init is None else init
    log = [] if log_last is None else log_last
    time_fw = 0.
    time_bw = 0.
    for epoch in range(num_step):
        # ts.grad = None
        start_fw = time.time()
        inside = inside_forward_stoc_log_torch(ts, device)
        end_fw= time.time()
        time_fw += end_fw - start_fw
        count = inside[0][l-1]
        print(f'step: {epoch: 4d}, log count: ', count.item())
        # print(count)
        if not count.requires_grad:
            log.append(count.item())
            break
        start_bw = time.time()
        count.backward()
        end_bw = time.time()
        time_bw += end_bw - start_bw        # print(ts.grad)
        ts_next = ts - ts.grad*lr
        ts_next = projection_simplex_torch_batch(ts_next.detach(), device=device)
        log.append(count.item())
        if torch.linalg.norm(ts_next-ts)/l < 1e-5:
            ts = ts_next
            break
        ts = ts_next
        ts.requires_grad_(True)
    print('optimal solution:')
    for i in range(l):
        print(f"A: {ts[i][0]:.4f}, C: {ts[i][1]:.4f}, G: {ts[i][2]:.4f}, U: {ts[i][3]:.4f}")
    print('optimal sequence:')
    seq = "".join([nucs[index] for index in torch.argmax(ts, axis=1)])
    print(seq)
    print('count:', inside_forward(seq)[0][l-1])
    return ts, log, time_fw, time_bw


def max_log_count(l, lr, num_step, init=None, log_last=None, device='cpu'):
    ts = torch.tensor(rand_arr(l), requires_grad=True, device=device) if init is None else init
    log = [] if log_last is None else log_last
    time_fw = 0.
    time_bw = 0.
    for epoch in range(num_step):
        ts.grad = None
        start_fw = time.time()
        inside = inside_forward_stoc_log_torch(ts, device)
        end_fw= time.time()
        time_fw += end_fw - start_fw
        count = -inside[0][l-1]
        print(f'step: {epoch: 4d}, log count: ', -count.item())
        start_bw = time.time()
        count.backward()
        end_bw = time.time()
        time_bw += end_bw - start_bw
        # print(ts.grad)
        ts_next = ts - ts.grad*lr
        ts_next = projection_simplex_torch_batch(ts_next.detach(), device=device)
        log.append(count.item())
        if torch.linalg.norm(ts_next-ts)/l < 1e-5:
            ts = ts_next
            break
        ts = ts_next
        ts.requires_grad_(True)
        # if len(log) > 2 and abs(log[-1]-log[-2])<1e-5:
        #     break
        # if (epoch+1)%100 == 0:
            # lr = lr/2
    print('optimal solution:')
    for i in range(l):
        print(f"A: {ts[i][0]:.4f}, C: {ts[i][1]:.4f}, G: {ts[i][2]:.4f}, U: {ts[i][3]:.4f}")
    print('optimal sequence:')
    seq = "".join([nucs[index] for index in torch.argmax(ts, axis=1)])
    print(seq)
    print('count:', inside_forward(seq)[0][l-1])
    return ts, log, time_fw, time_bw


def min_log_pt(l, lr, num_step, init=None, log_last=None, device='cpu'):
    ts = torch.tensor(rand_arr(l), requires_grad=True, device=device) if init is None else init
    log = [] if log_last is None else log_last
    time_fw = 0.
    time_bw = 0.
    for epoch in range(num_step):
        # ts.grad = None
        start_fw = time.time()
        inside = inside_forward_pt_stoc_log_torch(ts, device)
        end_fw= time.time()
        time_fw += end_fw - start_fw
        count = inside[0][l-1]
        print(f'step: {epoch: 4d}, log count: ', count.item())
        # print(count)
        if not count.requires_grad:
            log.append(count.item())
            break
        start_bw = time.time()
        count.backward()
        end_bw = time.time()
        time_bw += end_bw - start_bw
        # print(ts.grad)
        ts_next = ts - ts.grad*lr
        ts_next = projection_simplex_torch_batch(ts_next.detach(), device=device)
        log.append(count.item())
        if torch.linalg.norm(ts_next-ts)/l < 1e-5:
            ts = ts_next
            break
        ts = ts_next
        ts.requires_grad_(True)
    print('optimal solution:')
    for i in range(l):
        print(f"A: {ts[i][0]:.4f}, C: {ts[i][1]:.4f}, G: {ts[i][2]:.4f}, U: {ts[i][3]:.4f}")
    print('optimal sequence:')
    seq = "".join([nucs[index] for index in torch.argmax(ts, axis=1)])
    print(seq)
    print('count:', inside_forward_pt(seq)[0][l-1])
    return ts, log, time_fw, time_bw


def max_log_pt(l, lr, num_step, init=None, log_last=None, device='cpu'):
    ts = torch.tensor(rand_arr(l), requires_grad=True, device=device) if init is None else init
    log = [] if log_last is None else log_last
    time_fw = 0.
    time_bw = 0.
    for epoch in range(num_step):
        ts.grad = None
        start_fw = time.time()
        inside = inside_forward_pt_stoc_log_torch(ts, device)
        end_fw= time.time()
        time_fw += end_fw - start_fw
        count = -inside[0][l-1]
        print(f'step: {epoch: 4d}, log count: ', -count.item())
        start_bw = time.time()
        count.backward()
        end_bw = time.time()
        time_bw += end_bw - start_bw
        # print(ts.grad)
        ts_next = ts - ts.grad*lr
        ts_next = projection_simplex_torch_batch(ts_next.detach(), device=device)
        log.append(count.item())
        if torch.linalg.norm(ts_next-ts)/l < 1e-5:
            ts = ts_next
            break
        ts = ts_next
        ts.requires_grad_(True)
    print('optimal solution:')
    for i in range(l):
        print(f"A: {ts[i][0]:.4f}, C: {ts[i][1]:.4f}, G: {ts[i][2]:.4f}, U: {ts[i][3]:.4f}")
    print('optimal sequence:')
    seq = "".join([nucs[index] for index in torch.argmax(ts, axis=1)])
    print(seq)
    print('count:', inside_forward_pt(seq)[0][l-1])
    return ts, log, time_fw, time_bw


def run_batch(f, l_max, lr, num_step, device, folder="data/partition"):
    # results = dict()
    for l in range(5, l_max):
        filename = f'{f.__name__}_{l}_{lr:.2f}.json'
        if os.path.exists(os.path.join(folder, filename)):
            continue
        print(f'{f.__name__}: length={l}, lr={lr}, num_step={num_step}, device={device}, sharpturn={sharpturn}')
        d = dict()
        start = time.time()
        seq, log, tf, tb = f(l, lr, num_step, device=device)
        end = time.time()
        d['seq'] = seq.cpu().detach().numpy().tolist()
        d['log'] = log
        d['l'] = l
        d['lr'] = lr
        d['step'] = num_step
        d['device'] = device
        d['time'] = end - start
        d['time_fw'] = tf
        d['time_bw'] = tb
        # results[l] = d
        with open(os.path.join(folder, filename), 'w') as fw:
            json.dump(d, fw)
        

def run(f, l, lr, num_step, device):
    print(f'{f.__name__}: length={l}, lr={lr}, num_step={num_step}, device={device}, sharpturn={sharpturn}')
    start = time.time()
    seq, log, tf, tb = f(l, lr, num_step, device=device)
    end = time.time()
    import plotext as plt
    plt.plot(log)
    plt.plot_size(60, 20)
    plt.xlabel('step')
    ylabel = 'log count' if 'count' in f.__name__ else 'log partition'
    if  'max' in f.__name__:
        ylabel = 'negative ' + ylabel
    plt.ylabel(ylabel)
    plt.title(f'{f.__name__}: l={l}, step={len(log)}, time={end-start:.1f}')
    # plt.title(f'{f.__name__}: length={l}, lr={lr}, num_step={len(log)}, time={end-start:.1f} seconds, device={device}')
    print(f'time forward: {tf: .1f}, time backward: {tb: .1f}')
    plt.show()

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", '-l', type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--step", type=int, default=200)
    parser.add_argument("--device", '-d', type=str, default="cpu")
    parser.add_argument("--sharpturn", type=int, default=3)
    parser.add_argument("--min", action='store_true')
    parser.add_argument("--batch", '-b', action='store_true')
    parser.add_argument("--count", action='store_true')
    args = parser.parse_args()
    print('args:')
    print(args)
    set_sharpturn(args.sharpturn)
    
    l = args.length
    lr = args.lr
    num_step = args.step
    device = args.device
    if args.count:
        f = min_log_count if args.min else max_log_count
    else:
        f = min_log_pt if args.min else max_log_pt
    if args.batch:
        run_batch(f, l, lr, num_step, device)
    else:
        run(f, l, lr, num_step, device)    