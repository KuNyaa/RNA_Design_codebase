import os
import sys
import time
import argparse
import json
from collections import defaultdict

from partition_stoc import all_inside, rand_arr, pairs_index, nucs, inside_forward_stoc,inside_forward, inside_forward_pt, sharpturn, pair2score, unpair_score, score_stack, kT, inside_forward_stoc_pt, inside_forward_pt_stack, inside_forward_stoc_pt_stack

from partition_torch import inside_forward_stoc_log_torch

import numpy as np
import torch

import dynet as dy

# kT = 61.63207755
# kT = 1


def set_sharpturn(sharpturn_val):
    global sharpturn
    sharpturn = sharpturn_val
    print(f"sharpturn is: {sharpturn}")

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


def inside_forward_stoc_log_dy(s, device='cpu'): # s: n*4 matrix, row sum-to-one; A C G U
    # assert len(s) > 1, "the length of rna should be at least 2!"
    n = s.dim()[0][0]
    counts = defaultdict(lambda: defaultdict(lambda: dy.scalarInput(float('-inf')))) # number of structures
    for k in range(n):
        counts[k][k]= dy.scalarInput(0.)
        counts[k][k-1] = dy.scalarInput(0.)
    for j in range(1, n):
        for i in range(0, j):
            counts[i][j] = dy.logsumexp([counts[i][j], counts[i][j-1]]) # torch.logaddexp(counts[i][j], counts[i][j-1]) # x.
            if j-i>sharpturn:
                prob = dy.scalarInput(1e-32) # torch.tensor(0., device=device)
                for il, jr in pairs_index: 
                    prob += s[i][il]*s[j][jr]
                # if prob.value() > 0: # prob > 0
                prob = dy.log(prob) # torch.log(prob)
                counts_right = counts[i+1][j-1] 
                for t in range(0, i):
                    counts_left = counts[t][i-1] 
                    counts[t][j] = dy.logsumexp([counts[t][j], counts_left+counts_right+prob]) #  torch.logaddexp(counts[t][j], counts_left+counts_right+prob) # x(x)
                counts[i][j] = dy.logsumexp([counts[i][j], counts_right+prob])  # torch.logaddexp(counts[i][j], counts_right+prob) # (x)
    return counts


def inside_forward_pt_stoc_log_dy(s, device='cpu'): # s: n*4 matrix, row sum-to-one; A C G U
    # assert len(s) > 1, "the length of rna should be at least 2!"
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
    return counts


def outside_forward_stoc_pt_dy(s, inside): # s: n*4 matrix, row sum-to-one; A C G U
    INF_NEG = '-1e32'
    # assert len(s) > 1, "the length of rna should be at least 2!"
    # print(f'sharpturn: {sharpturn}')
    n = s.dim()[0][0]
    outside = defaultdict(lambda: defaultdict(lambda: dy.scalarInput(float(INF_NEG)))) # number of structures
    outside[0][n-1] = dy.scalarInput(0.)
    marginal = defaultdict(lambda: defaultdict(lambda: dy.scalarInput(float(INF_NEG))))
    for j in range(n-1, -1, -1):
        for i in range(j-1, -1, -1):
            # counts[i][j] = np.logaddexp(counts[i][j], counts[i][j-1]+(-unpair_score/kT))  # x.
            outside[i][j-1] = dy.logsumexp([outside[i][j-1], outside[i][j]+(-unpair_score/kT)])
            if j-i>sharpturn:
                score_ij = dy.scalarInput(float(INF_NEG))
                for il, jr in pairs_index: 
                    prob = s[i][il]*s[j][jr] + dy.scalarInput(float(1e-32))
                    pair_ij = nucs[il]+nucs[jr]
                    score_ij = dy.logsumexp([score_ij, dy.log(prob) + (-pair2score[pair_ij]/kT)]) 
                for t in range(0, i+1):
                    outside[t][i-1] = dy.logsumexp([outside[t][i-1], outside[t][j]+inside[i+1][j-1]+score_ij])
                    outside[i+1][j-1] = dy.logsumexp([outside[i+1][j-1], outside[t][j]+inside[t][i-1]+score_ij])
                    marginal[i][j] = dy.logsumexp([marginal[i][j], outside[t][j]+inside[t][i-1]+inside[i+1][j-1]+score_ij-inside[0][n-1]])
    return outside, marginal


def inside_forward_pt_stack_stoc_log_dy(s, device='cpu'):
    # assert len(s) > 1, "the length of rna should be at least 2!"
    INF_NEG = '-1e32'
    n = s.dim()[0][0]
    p = defaultdict(lambda: defaultdict(lambda: dy.scalarInput(float(INF_NEG)))) # partition
    a = defaultdict(lambda: defaultdict(lambda: dy.scalarInput(float(INF_NEG)))) # left and right don't pair
    b = defaultdict(lambda: defaultdict(lambda: dy.scalarInput(float(INF_NEG)))) # left and right pair
    for k in range(n):
        p[k][k]= dy.scalarInput(-unpair_score/kT)
        p[k][k-1] = dy.scalarInput(0.)
        a[k][k]= dy.scalarInput(-unpair_score/kT)
        a[k][k-1] = dy.scalarInput(0.)      
    for j in range(1, n):
        for i in range(0, j):
            a[i][j] = dy.logsumexp([a[i][j], p[i][j-1]+(-unpair_score/kT)])  # a <- s.
            p[i][j] = dy.logsumexp([p[i][j], p[i][j-1]+(-unpair_score/kT)])  # p <- s.
            if j-i > sharpturn:
                score_ij = dy.scalarInput(float(INF_NEG))
                score_ij_stack = dy.scalarInput(float(INF_NEG))
                for il, jr in pairs_index:
                    prob = s[i][il]*s[j][jr] + dy.scalarInput(float(1e-32))
                    # if prob > 0:
                    pair_ij = nucs[il]+nucs[jr]
                    score_ij = dy.logsumexp([score_ij, dy.log(prob) + (-pair2score[pair_ij]/kT)])
                    score_ij_stack = dy.logsumexp([score_ij_stack, dy.log(prob) + (-pair2score[pair_ij]/kT) + (-score_stack/kT)])
                a_right = a[i+1][j-1]
                b_right = b[i+1][j-1]
                for t in range(0, i):
                    p_left = p[t][i-1] 
                    a[t][j] = dy.logsumexp([a[t][j], p_left+a_right+score_ij]) # a <- s(a)
                    p[t][j] = dy.logsumexp([p[t][j], p_left+a_right+score_ij]) # p <- s(a)
                    a[t][j] = dy.logsumexp([a[t][j], p_left+b_right+score_ij_stack]) # a <- s(b)
                    p[t][j] = dy.logsumexp([p[t][j], p_left+b_right+score_ij_stack]) # p <- s(b)
                b[i][j] = dy.logsumexp([b[i][j], a_right+score_ij]) # b <- (a)
                p[i][j] = dy.logsumexp([p[i][j], a_right+score_ij]) # p <- (a)
                b[i][j] = dy.logsumexp([b[i][j], b_right+score_ij_stack]) # b <- (b)
                p[i][j] = dy.logsumexp([p[i][j], b_right+score_ij_stack]) # p <- (b)
    return p


def max_log_count_dy_dn(l, lr, num_step, init=None, log_last=None, device='cpu'):
    m = dy.ParameterCollection()
    ts = m.add_parameters((l, 4))
    if init is None:
        ts.set_value(rand_arr(l))
    else:
        ts.set_value(init)
    # build computational graph
    # dy.renew_cg(immediate_compute=True)
    # x = dy.scalarInput(0.)
    ts_input = ts #+ x
    inside = inside_forward_stoc_log_dy(ts_input, device)
    # set records
    log = [] if log_last is None else log_last
    time_fw = 0.
    time_bw = 0.
    for epoch in range(num_step):
        start_fw = time.time()
        # x.set(0.)
        dy.renew_cg(immediate_compute=True)
        # x = dy.scalarInput(0.)
        ts_input = ts #+ x
        inside = inside_forward_stoc_log_dy(ts_input, device)
        count = -inside[0][l-1]
        print(f'step: {epoch: 4d}, log count: ', -count.value())
        end_fw= time.time()
        time_fw += end_fw - start_fw
        start_bw = time.time()
        count.backward()
        end_bw = time.time()
        time_bw += end_bw - start_bw
        print(f'time_fw: {end_fw - start_fw}, time_bw: {end_bw - start_bw}')
        # print(ts.grad)
        ts_next = ts.value() - lr*ts.gradient()
        ts_next = projection_simplex_np_batch(ts_next)
        log.append(count.value())
        norm_diff = np.linalg.norm(ts.value()-ts_next)/l
        if norm_diff < 1e-6:
            # norm_grad = np.linalg.norm(ts.gradient())
            ts.set_value(ts_next)
            break
        # norm_grad = np.linalg.norm(ts.gradient())
        ts.set_value(ts_next)
    print('optimal solution:')
    ts = ts.value()
    for i in range(l):
        print(f"A: {ts[i][0]:.4f}, C: {ts[i][1]:.4f}, G: {ts[i][2]:.4f}, U: {ts[i][3]:.4f}")
    print('optimal sequence:')
    seq = "".join([nucs[index] for index in np.argmax(ts, axis=1)])
    print(seq)
    # print(f'grad: {norm_grad:.8e}')
    print(f'diff: {norm_diff:.8e}')
    print('count:', inside_forward(seq)[0][l-1])
    return ts, log, time_fw, time_bw


def max_log_count_dy(l, lr, num_step, init=None, log_last=None, device='cpu'):
    m = dy.ParameterCollection()
    ts = m.add_parameters((l, 4))
    if init is None:
        ts.set_value(rand_arr(l))
    else:
        ts.set_value(init)
    # build computational graph
    dy.renew_cg()
    x = dy.scalarInput(0.)
    ts_input = ts + x
    time_build = time.time()
    inside = inside_forward_stoc_log_dy(ts_input, device)
    time_graph = time.time() - time_build
    print(f'time to build graph: {time_graph}')
    # set records
    log = [] if log_last is None else log_last
    time_fw = 0.
    time_bw = 0.
    for epoch in range(num_step):
        start_fw = time.time()
        x.set(0.)
        if epoch == 0:
            time_set = time.time() - start_fw
            print(f'first set: {time_set}')
        count = -inside[0][l-1]
        print(f'step: {epoch: 4d}, log count: ', -count.value())
        end_fw= time.time()
        time_fw += end_fw - start_fw
        start_bw = time.time()
        count.backward()
        end_bw = time.time()
        time_bw += end_bw - start_bw
        print(f'time_fw: {end_fw - start_fw}, time_bw: {end_bw - start_bw}')
        # print(ts.grad)
        ts_next = ts.value() - lr*ts.gradient()
        ts_next = projection_simplex_np_batch(ts_next)
        log.append(count.value())
        norm_diff = np.linalg.norm(ts.value()-ts_next)/l
        if norm_diff < 1e-6:
            # norm_grad = np.linalg.norm(ts.gradient())
            ts.set_value(ts_next)
            break
        # norm_grad = np.linalg.norm(ts.gradient())
        ts.set_value(ts_next)
    print('optimal solution:')
    ts = ts.value()
    for i in range(l):
        print(f"A: {ts[i][0]:.4f}, C: {ts[i][1]:.4f}, G: {ts[i][2]:.4f}, U: {ts[i][3]:.4f}")
    print('optimal sequence:')
    seq = "".join([nucs[index] for index in np.argmax(ts, axis=1)])
    print(seq)
    # print(f'grad: {norm_grad:.8e}')
    print(f'diff: {norm_diff:.8e}')
    print('count:', inside_forward(seq)[0][l-1])
    return ts, log, time_fw, time_bw, time_graph, time_set


def min_log_count_dy(l, lr, num_step, init=None, log_last=None, device='cpu'):
    m = dy.ParameterCollection()
    ts = m.add_parameters((l, 4))
    if init is None:
        ts.set_value(rand_arr(l))
    else:
        ts.set_value(init)
    # build computational graph
    dy.renew_cg()
    x = dy.scalarInput(0.)
    ts_input = ts + x
    inside = inside_forward_stoc_log_dy(ts_input, device)
    # set records
    log = [] if log_last is None else log_last
    time_fw = 0.
    time_bw = 0.
    for epoch in range(num_step):
        start_fw = time.time()
        x.set(0.)
        count = inside[0][l-1]
        print(f'step: {epoch: 4d}, log count: ', -count.value())
        end_fw= time.time()
        time_fw += end_fw - start_fw
        start_bw = time.time()
        count.backward()
        end_bw = time.time()
        time_bw += end_bw - start_bw
        # print(ts.grad)
        ts_next = ts.value() - lr*ts.gradient()
        ts_next = projection_simplex_np_batch(ts_next)
        log.append(count.value())
        norm_diff = np.linalg.norm(ts.value()-ts_next)/l
        if norm_diff< 1e-6:
            ts.set_value(ts_next)
            break
        ts.set_value(ts_next)
    print('optimal solution:')
    ts = ts.value()
    for i in range(l):
        print(f"A: {ts[i][0]:.4f}, C: {ts[i][1]:.4f}, G: {ts[i][2]:.4f}, U: {ts[i][3]:.4f}")
    print('optimal sequence:')
    seq = "".join([nucs[index] for index in np.argmax(ts, axis=1)])
    print(seq)
    print(f'diff: {norm_diff:.8e}')
    print('count:', inside_forward(seq)[0][l-1])
    return ts, log, time_fw, time_bw


def test_set_dy(l, init=None):
    m = dy.ParameterCollection()
    ts = m.add_parameters((l, 4))
    print('intial by dynet:')
    # print(ts.value())
    if init is None:
        value = rand_arr(l)
        print('target:')
        print(value)
        ts.set_value(value)
    else:
        ts.set_value(init)
    print('final:')
    print(ts.value())


def max_log_pt_dy(l, lr, num_step, init=None, log_last=None, device='cpu'):
    m = dy.ParameterCollection()
    ts = m.add_parameters((l, 4))
    if init is None:
        ts.set_value(rand_arr(l))
    else:
        ts.set_value(init)
    # build computational graph
    dy.renew_cg()
    x = dy.scalarInput(0.)
    ts_input = ts + x
    inside = inside_forward_pt_stoc_log_dy(ts_input, device)
    # set records
    log = [] if log_last is None else log_last
    time_fw = 0.
    time_bw = 0.
    for epoch in range(num_step):
        start_fw = time.time()
        x.set(0.)
        count = -inside[0][l-1]
        print(f'step: {epoch: 4d}, log partition: ', -count.value())
        end_fw= time.time()
        time_fw += end_fw - start_fw
        start_bw = time.time()
        count.backward()
        end_bw = time.time()
        time_bw += end_bw - start_bw
        # print(ts.grad)
        ts_next = ts.value() - lr*ts.gradient()
        ts_next = projection_simplex_np_batch(ts_next)
        log.append(count.value())
        norm_diff = np.linalg.norm(ts.value()-ts_next)/l
        if norm_diff < 1e-6:
            ts.set_value(ts_next)
            break
        ts.set_value(ts_next)
    print('optimal solution:')
    ts = ts.value()
    for i in range(l):
        print(f"A: {ts[i][0]:.4f}, C: {ts[i][1]:.4f}, G: {ts[i][2]:.4f}, U: {ts[i][3]:.4f}")
    print('optimal sequence:')
    seq = "".join([nucs[index] for index in np.argmax(ts, axis=1)])
    print(seq)
    print(f'diff: {norm_diff:.8e}')
    print('pt_dy: ', -count.value())
    pt_stoc = inside_forward_stoc_pt(ts)
    print('pt_pt: ', pt_stoc[0][l-1])
    # import torch
    # pt_torch = inside_forward_stoc_log_torch(torch.tensor(ts))
    # print('pt torch: ', pt_torch[0][l-1])
    print('partition:', inside_forward_pt(seq)[0][l-1])
    return ts, log, time_fw, time_bw


def min_log_pt_dy(l, lr, num_step, init=None, log_last=None, device='cpu'):
    m = dy.ParameterCollection()
    ts = m.add_parameters((l, 4))
    if init is None:
        ts.set_value(rand_arr(l))
    else:
        ts.set_value(init)
    # build computational graph
    dy.renew_cg()
    x = dy.scalarInput(0.)
    ts_input = ts + x
    inside = inside_forward_pt_stoc_log_dy(ts_input, device)
    # set records
    log = [] if log_last is None else log_last
    time_fw = 0.
    time_bw = 0.
    for epoch in range(num_step):
        start_fw = time.time()
        x.set(0.)
        count = inside[0][l-1]
        print(f'step: {epoch: 4d}, log partition: ', count.value())
        end_fw= time.time()
        time_fw += end_fw - start_fw
        start_bw = time.time()
        count.backward()
        end_bw = time.time()
        time_bw += end_bw - start_bw
        # print(ts.grad)
        ts_next = ts.value() - lr*ts.gradient()
        ts_next = projection_simplex_np_batch(ts_next)
        log.append(count.value())
        norm_diff = np.linalg.norm(ts.value()-ts_next)/l
        if norm_diff < 1e-6:
            ts.set_value(ts_next)
            break
        ts.set_value(ts_next)
    print('optimal solution:')
    ts = ts.value()
    for i in range(l):
        print(f"A: {ts[i][0]:.4f}, C: {ts[i][1]:.4f}, G: {ts[i][2]:.4f}, U: {ts[i][3]:.4f}")
    print('optimal sequence:')
    seq = "".join([nucs[index] for index in np.argmax(ts, axis=1)])
    print(seq)
    print(f'diff: {norm_diff:.8e}')
    pt_stoc = inside_forward_stoc_pt(ts)
    print('pt stoc: ', pt_stoc[0][l-1])
    import torch
    pt_torch = inside_forward_stoc_log_torch(torch.tensor(ts))
    print('pt torch: ', pt_torch[0][l-1])
    print('partition:', inside_forward_pt(seq)[0][l-1])
    return ts, log, time_fw, time_bw


def base_pair_prob(n, marginal):
    bpp = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            pr = dy.exp(marginal[i][j]).value()
            print(i, j, pr)
            bpp[i][j] = pr
    bpp += bpp.T
    unpair = 1 - np.sum(bpp, axis=1)
    bpp[range(len(bpp)), range(len(bpp))] = unpair
    return bpp

def defect(ss, bpp):
    from utils.structure import extract_pairs
    pairs = extract_pairs(ss)
    defect_pos = [1 - bpp[i, j] for i, j in enumerate(pairs)]
    return sum(defect_pos)

def margin2defect(ss, marginal):
    from utils.structure import extract_pairs
    pairs = extract_pairs(ss)
    defect = 0
    for i in range(len(ss)):
        if pairs[i] == i:
            for j in range(i+1, len(ss)):
                defect += dy.exp(marginal[i][j])
            for k in range(i):
                defect += dy.exp(marginal[k][i])
        elif pairs[i] < i:
            defect += 1 - dy.exp(marginal[pairs[i]][i])
        else:
            defect += 1 - dy.exp(marginal[i][pairs[i]])
    return defect/len(ss)

def min_log_ned_dy(ss, lr, num_step, init=None, log_last=None, device='cpu'):
    l = len(ss)
    m = dy.ParameterCollection()
    ts = m.add_parameters((l, 4))
    if init is None:
        ts.set_value(rand_arr(l))
    else:
        ts.set_value(init)
    # build computational graph
    dy.renew_cg()
    x = dy.scalarInput(0.)
    ts_input = ts + x
    inside = inside_forward_pt_stoc_log_dy(ts_input, device)
    outside, marginal = outside_forward_stoc_pt_dy(ts_input, inside)
    ned = margin2defect(ss, marginal)
    # set records
    log = [] if log_last is None else log_last
    time_fw = 0.
    time_bw = 0.
    for epoch in range(num_step):
        start_fw = time.time()
        x.set(0.)
        # count = -inside[0][l-1]
        # count = marginal[4][8]
        print(f'step: {epoch: 4d}, ned: ', ned.value())
        end_fw= time.time()
        time_fw += end_fw - start_fw
        start_bw = time.time()
        ned.backward()
        end_bw = time.time()
        time_bw += end_bw - start_bw
        # print(ts.grad)
        ts_next = ts.value() - lr*ts.gradient()
        ts_next = projection_simplex_np_batch(ts_next)
        log.append(ned.value())
        norm_diff = np.linalg.norm(ts.value()-ts_next)/l
        if norm_diff < 1e-12:
            ts.set_value(ts_next)
            break
        ts.set_value(ts_next)
    print('optimal solution:')
    ts = ts.value()
    ts_list = list(ts.flatten())
    for i in range(l):
        print(f"A: {ts[i][0]:.4f}, C: {ts[i][1]:.4f}, G: {ts[i][2]:.4f}, U: {ts[i][3]:.4f}")
    print('optimal sequence:')
    seq = "".join([nucs[index] for index in np.argmax(ts, axis=1)])
    print(ss)
    print(seq)
    print(f'diff: {norm_diff:.8e}')
    return ts_list, seq, log, time_fw, time_bw


def ned_txt(path, name_pz, lr, num_step):
    import pandas as pd
    target_list = []
    with open(path) as f:
        for line in f:
            target_list.append(line.strip())
    data = []
    for i, target in enumerate(target_list):
        puzzle = f"{name_pz}_{i}"
        print(i, puzzle)
        print('target: ', target)
        start_time = time.time()
        ts_list, seq, log, time_fw, time_bw = min_log_ned_dy(target, lr, num_step, init=None, log_last=None, device='cpu')
        finish_time = time.time()
        p = seq
        v = log[-1]
        print('RNA: ', p)
        print('ensemble defect: ', v)
        print(target)
        # ss_mfe = mfe(p)[0]
        # dist = struct_dist(target, ss_mfe)
        # print(ss_mfe)
        # print(f'structure distance: {dist}')
        # print('mfe_list: ', mfe_list)
        data.append([puzzle, target, p, v, time_fw+time_bw, time_fw, time_bw, log, ts_list])
        filename = f"{name_pz}_sgd_step{num_step}.csv"
        # if i == 1:
        #     break
        if i >= 0:
            df_arw = pd.DataFrame(data, columns=('puzzle_name', 'structure', 'rna', 'objective', 'time', 'time_fw', 'time_bw', 'log', 'rna_arr'))
            df_arw.to_csv(filename)
    df_arw = pd.DataFrame(data, columns=('puzzle_name', 'structure', 'rna', 'objective', 'time', 'time_fw', 'time_bw', 'log', 'rna_arr'))
    df_arw.to_csv(filename)
    


def max_log_pt_stack_dy(l, lr, num_step, init=None, log_last=None, device='cpu'):
    m = dy.ParameterCollection()
    ts = m.add_parameters((l, 4))
    if init is None:
        ts.set_value(rand_arr(l))
    else:
        ts.set_value(init)
    # build computational graph
    dy.renew_cg()
    x = dy.scalarInput(0.)
    ts_input = ts + x
    inside = inside_forward_pt_stack_stoc_log_dy(ts_input, device)
    # set records
    log = [] if log_last is None else log_last
    time_fw = 0.
    time_bw = 0.
    for epoch in range(num_step):
        start_fw = time.time()
        x.set(0.)
        count = -inside[0][l-1]
        objective =  -count.value()
        end_fw= time.time()
        time_fw += end_fw - start_fw
        start_bw = time.time()
        count.backward()
        end_bw = time.time()
        time_bw += end_bw - start_bw
        # print(f'grad: {np.linalg.norm(ts.gradient())}')
        ts_next = ts.value() - lr*ts.gradient()
        ts_next = projection_simplex_np_batch(ts_next)
        log.append(count.value())
        norm_diff = np.linalg.norm(ts.value()-ts_next)/l
        print(f'step: {epoch: 4d}, log partition: {objective: .8f}, grad: {np.linalg.norm(ts.gradient()): 6e}, diff: {norm_diff: .6e}')
        # if num_step - epoch < 10:
        #     print(f'diff: {norm_diff: .8e}')
        if norm_diff < 1e-9:
            ts.set_value(ts_next)
            break
        ts.set_value(ts_next)
    print(ts.gradient())
    print('optimal solution:')
    ts = ts.value()
    for i in range(l):
        print(f"A: {ts[i][0]:.4f}, C: {ts[i][1]:.4f}, G: {ts[i][2]:.4f}, U: {ts[i][3]:.4f}")
    print('optimal sequence:')
    seq = "".join([nucs[index] for index in np.argmax(ts, axis=1)])
    print(seq)
    print(f'diff: {norm_diff:.8e}')
    print('pt_dy: ', -count.value())
    pt_stoc, a, b = inside_forward_stoc_pt_stack(ts)
    print('pt_pt: ', pt_stoc[0][l-1])
    # print('partition:', inside_forward_pt_stack(seq)[0][0][l-1])
    return ts, log, time_fw, time_bw



def run_batch(f, l_max, lr, num_step, device, folder="data/partition_dy"):
    # results = dict()
    for l in range(5, l_max):
        filename = f'{f.__name__}_{l}_{lr:.2f}.json'
        if os.path.exists(os.path.join(folder, filename)):
            continue
        print(f'{f.__name__}: length={l}, lr={lr}, num_step={num_step}, device={device}, sharpturn={sharpturn}')
        d = dict()
        start = time.time()
        if f == max_log_count_dy:
            seq, log, tf, tb, tg, ts = f(l, lr, num_step, device=device)
        else:
            seq, log, tf, tb = f(l, lr, num_step, device=device)
        end = time.time()
        d['seq'] = seq.tolist() # .cpu().detach().numpy().tolist()
        d['log'] = log
        d['l'] = l
        d['lr'] = lr
        d['step'] = num_step
        d['device'] = device
        d['time'] = end - start
        d['time_fw'] = tf
        d['time_bw'] = tb
        if f == max_log_count_dy:
            d['time_graph'] = tg
            d['time_set'] = ts
        # results[l] = d
        with open(os.path.join(folder, filename), 'w') as fw:
            json.dump(d, fw)
        

def run(f, l, lr, num_step, device):
    print(f'{f.__name__}: length={l}, lr={lr}, num_step={num_step}, device={device}, sharpturn={sharpturn}')
    start = time.time()
    if f == max_log_count_dy:
        seq, log, tf, tb, tg, ts = f(l, lr, num_step, device=device)
    else:
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
    if f == max_log_count_dy:
        print(f'time graph  : {tg}')
        print(f'time set : {ts}')
    print(f'sharpturn: {sharpturn}')
    plt.show()
    
    
def valid(l):
    init = None
    m = dy.ParameterCollection()
    ts = m.add_parameters((l, 4))
    if init is None:
        ts.set_value(rand_arr(l))
    else:
        ts.set_value(init)
    # build computational graph
    dy.renew_cg()
    x = dy.scalarInput(0.)
    ts_input = ts + x
    inside = inside_forward_pt_stoc_log_dy(ts_input, 'cpu')
    x.set(0.)
    pt_dy = inside[0][l-1].value()
    print(f'pt_dy: {pt_dy}')
    from partition_stoc import inside_forward_stoc_pt
    pt_np = inside_forward_stoc_pt(ts_input.value())
    print(f'pt_np: {pt_np[0][l-1]}')
    import torch
    from partition_torch import inside_forward_pt_stoc_log_torch
    pt_torch = inside_forward_pt_stoc_log_torch(torch.tensor(ts_input.value()))
    print(f'pt_torch: {pt_torch[0][l-1].item()}')


        
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
    parser.add_argument("--stack", action='store_true')
    parser.add_argument('--valid', '-v', action='store_true')
    parser.add_argument('--folder', type=str, default='data/partition_dy_time')
    parser.add_argument('--dn', action='store_true')
    parser.add_argument('--ned', action='store_true')
    parser.add_argument('--txt', action='store_true')
    parser.add_argument('--path', type=str, default="data/eterna100/eterna100.txt")
    parser.add_argument('--name', type=str, default="")
    parser.add_argument('--dy', action='store_true')
    args = parser.parse_args()
    print('args:')
    print(args)
    set_sharpturn(args.sharpturn)
    
    l = args.length
    lr = args.lr
    num_step = args.step
    device = args.device
    if args.dy:
        test_set_dy(args.length)
        exit(0)
    if args.valid:
        valid(l)
        exit(0)
    if args.txt:
        if not args.name:
            args.name = args.path.split("/")[-1].split('.')[0]
        ned_txt(args.path, args.name, args.lr, args.step)
    if args.ned:
        ss = '((....)).((....))'
        max_log_ned_dy(ss=ss, lr=args.lr, num_step=args.step, init=None, log_last=None, device='cpu')
        exit(0)
    if args.stack:
        f = max_log_pt_stack_dy
    else:
        if args.count:
            f = min_log_count_dy if args.min else max_log_count_dy
        else:
            f = min_log_pt_dy if args.min else max_log_pt_dy
        if args.dn:
            f = max_log_count_dy_dn
    if args.batch:
        run_batch(f, l, lr, num_step, device, args.folder)
    else:
        run(f, l, lr, num_step, device)  