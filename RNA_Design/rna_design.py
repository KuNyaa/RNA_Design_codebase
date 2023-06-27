import os
import sys
import time
import argparse
import json
from collections import defaultdict

from partition_stoc import all_inside, rand_arr, inside_forward_stoc,inside_forward, inside_forward_pt, score_stack, inside_forward_stoc_pt, inside_forward_pt_stack, inside_forward_stoc_pt_stack

# from partition_torch import inside_forward_stoc_log_torch

import numpy as np
# import torch

import dynet as dy

nucs = 'ACGU'
pairs_index = [(0, 3), (3, 0), (1, 2), (2, 1), (2, 3), (3, 2)] 
unpair_index = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 3), (2, 0), (2, 2), (3, 1), (3, 3)]
pair2score = {"CG": -3, "GC": -3, "AU": -2, "UA":-2, "GU": -1, "UG":-1}
unpair_score = 1.

# kT = 61.63207755
kT = 1.

def projection_simplex_np_batch(x, z=1):
    x_sorted = np.sort(x, axis=1)[:, ::-1]
    cumsum = np.cumsum(x_sorted, axis=1)
    denom = np.arange(x.shape[1]) + 1
    theta = (cumsum - z)/denom
    mask = x_sorted > theta 
    csum_mask = np.cumsum(mask, axis=1)
    index = csum_mask[:, -1] - 1
    # print(f"index.shape: {index.shape}, theta.shape: {theta.shape}")
    x_proj = np.maximum(x.transpose() - theta[np.arange(len(x)), index], 0)
    return x_proj.transpose()

def inside_forward_pt_stoc_log_dy(s, struct, sharpturn, device='cpu'): # s: n*4 matrix, row sum-to-one; A C G U
    # assert len(s) > 1, "the length of rna should be at least 2!"

    # compute log Q hat
    n = s.dim()[0][0]
    counts = defaultdict(lambda: defaultdict(lambda: dy.scalarInput(float('-1e32')))) # number of structures
    for k in range(n):
        counts[k][k]= dy.scalarInput(-unpair_score/kT)
        counts[k][k-1] = dy.scalarInput(0.)
    for j in range(1, n):
        for i in range(0, j):
            counts[i][j] = dy.logsumexp([counts[i][j], counts[i][j-1]+(-unpair_score/kT)]) #torch.logaddexp(counts[i][j], counts[i][j-1]+(-unpair_score/kT)) # x.
            if j-i>sharpturn:
                score_ij = dy.scalarInput(float('-1e32')) #torch.tensor(-torch.inf, device=device)
                # prob_sum = dy.scalarInput(1e-32) #prob_sum = 0
                for il, jr in pairs_index: 
                    prob = s[i][il]*s[j][jr] + dy.scalarInput(1e-32)
                    # prob_sum += prob
                    # if prob > 0:
                    pair_ij = nucs[il]+nucs[jr]
                    score_ij = dy.logsumexp([score_ij, dy.log(prob)+(-pair2score[pair_ij]/kT)]) #torch.logaddexp(score_ij, torch.log(prob)+(-pair2score[pair_ij]/kT))
                # if prob_sum > 0:
                counts_right = counts[i+1][j-1] 
                for t in range(0, i):
                    counts_left = counts[t][i-1] 
                    counts[t][j] = dy.logsumexp([counts[t][j], counts_left+counts_right+score_ij]) # torch.logaddexp(counts[t][j], counts_left+counts_right+score_ij) # x(x)
                counts[i][j] = dy.logsumexp([counts[i][j], counts_right+score_ij]) #torch.logaddexp(counts[i][j], counts_right+score_ij) # (x)

    # count[0][n-1] is log E[Q hat]
    # compute E[delta G(X, y) / RT] and add to count[0][n-1]

    delta_G = dy.scalarInput(0.)

    stack = []
    for j in range(n):
        if struct[j] == '(':
            stack.append(j)
        elif struct[j] == ')':
            i = stack.pop()

            score_ij = dy.scalarInput(0.)
            
            for il, jr in pairs_index:
                pair_ij = nucs[il]+nucs[jr]
                prob = s[i][il]*s[j][jr]
                score_ij += prob * pair2score[pair_ij]

            for il, jr in unpair_index:
                prob = s[i][il]*s[j][jr]
                score_ij += prob * dy.scalarInput(100.)

            delta_G += score_ij 
        else:
            delta_G += unpair_score

    delta_G /= kT
    counts[0][n-1] += delta_G

    return counts

def min_objective(struct, lr, num_step, sharpturn, init=None, log_last=None, device='cpu'):
    l = len(struct)
    m = dy.ParameterCollection()
    ts = m.add_parameters((l, 4))

    if init is None:
        # ts.set_value(rand_arr(l))
        ts.set_value(np.full((l, 4), .25)) # Initialize all prob to .25
    else:
        ts.set_value(init)

    # build computational graph
    dy.renew_cg()
    x = dy.scalarInput(0.)
    ts_input = ts + x
    inside = inside_forward_pt_stoc_log_dy(ts_input, struct, sharpturn, device)

    # set records
    log = [] if log_last is None else log_last
    time_fw = 0.
    time_bw = 0.

    for epoch in range(num_step):
        start_fw = time.time()
        x.set(0.)

        # inside = inside_forward_pt_stoc_log_dy(ts, struct, device)
        count = inside[0][l-1]
        print(f'step: {epoch: 4d}, log partition: ', count.value())

        end_fw= time.time()
        time_fw += end_fw - start_fw
        start_bw = time.time()

        count.backward()

        end_bw = time.time()
        time_bw += end_bw - start_bw

        ts_next = ts.value() - lr*ts.gradient()
        ts_next = projection_simplex_np_batch(ts_next)

        log.append(count.value())

        norm_diff = np.linalg.norm(ts.value()-ts_next)/l
        ts.set_value(ts_next)
        if norm_diff < 1e-10:
            break
    

    print('optimal solution:')
    ts = ts.value()
    for i in range(l):
        print(f"A: {ts[i][0]:.4f}, C: {ts[i][1]:.4f}, G: {ts[i][2]:.4f}, U: {ts[i][3]:.4f}")
    
    print('structure:')
    print(struct)

    seq = "".join([nucs[index] for index in np.argmax(ts, axis=1)])
    print('optimal sequence:')
    print(seq)
    
    print(f'diff: {norm_diff:.8e}')
    
    pt_stoc = inside_forward_stoc_pt(ts)
    print('pt_dy: ', count.value())
    print('pt_pt: ', pt_stoc[0][l-1])
    print('partition:', inside_forward_pt(seq)[0][l-1])
    
    return ts, log, time_fw, time_bw

def run(f, s, lr, num_step, sharpturn, device):
    print(f'{f.__name__}: length={len(s)}, lr={lr}, num_step={num_step}, device={device}, sharpturn={sharpturn}')
    
    start = time.time()
    seq, log, tf, tb = f(s, lr, num_step, sharpturn, device=device)
    end = time.time()
    
    import plotext as plt
    plt.plot(log)
    plt.plot_size(60, 20)
    plt.xlabel('step')
    ylabel = 'log count' if 'count' in f.__name__ else 'log partition'
    
    if  'max' in f.__name__:
        ylabel = 'negative ' + ylabel
    
    plt.ylabel(ylabel)
    plt.title(f'{f.__name__}: l={len(s)}, step={len(log)}, time={end-start:.1f}')
    
    print(f'time forward: {tf: .1f}, time backward: {tb: .1f}')
    
    print(f'sharpturn: {sharpturn}')

    plt.show()

if __name__ == '__main__':
    # assume struct is a valid bracket structure for now
    # python rna_design.py -s '(((...)))' --lr 0.001 --step 10000 --sharpturn 0

    parser = argparse.ArgumentParser()
    parser.add_argument("--structure", '-s', type=str, default=".")
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--step", type=int, default=200)
    parser.add_argument("--sharpturn", type=int, default=0)
    parser.add_argument("--device", '-d', type=str, default="cpu")
    args = parser.parse_args()

    s = args.structure
    lr = args.lr
    num_step = args.step
    sharpturn = args.sharpturn
    device = args.device


    f = min_objective
    run(f, s, lr, num_step, sharpturn, device)
