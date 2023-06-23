import os
import sys
import time
import argparse
import json
from collections import defaultdict

from partition_stoc import all_inside, rand_arr, pairs_index, nucs, inside_forward_stoc,inside_forward, inside_forward_pt, sharpturn, set_sharpturn, pair2score, unpair_score, kT
from partition_dy import inside_forward_pt_stoc_log_dy, inside_forward_stoc_log_dy
from utils.structure import extract_pairs_list

import numpy as np
import pandas as pd
import torch

import dynet as dy

# kT = 61.63207755
# kT = 1

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


# def inside_forward_stoc_log_dy(s, device='cpu'): # s: n*4 matrix, row sum-to-one; A C G U
#     # assert len(s) > 1, "the length of rna should be at least 2!"
#     n = s.dim()[0][0]
#     counts = defaultdict(lambda: defaultdict(lambda: dy.scalarInput(float('-inf')))) # number of structures
#     for k in range(n):
#         counts[k][k]= dy.scalarInput(0.)
#         counts[k][k-1] = dy.scalarInput(0.)
#     for j in range(1, n):
#         for i in range(0, j):
#             counts[i][j] = dy.logsumexp([counts[i][j], counts[i][j-1]]) # torch.logaddexp(counts[i][j], counts[i][j-1]) # x.
#             if j-i>sharpturn:
#                 prob = dy.scalarInput(1e-32) # torch.tensor(0., device=device)
#                 for il, jr in pairs_index: 
#                     prob += s[i][il]*s[j][jr]
#                 if prob.value() > 0: # prob > 0
#                     prob = dy.log(prob) # torch.log(prob)
#                     counts_right = counts[i+1][j-1] 
#                     for t in range(0, i):
#                         counts_left = counts[t][i-1] 
#                         counts[t][j] = dy.logsumexp([counts[t][j], counts_left+counts_right+prob]) #  torch.logaddexp(counts[t][j], counts_left+counts_right+prob) # x(x)
#                     counts[i][j] = dy.logsumexp([counts[i][j], counts_right+prob])  # torch.logaddexp(counts[i][j], counts_right+prob) # (x)
#     return counts


# def inside_forward_pt_stoc_log_dy(s, device='cpu'): # s: n*4 matrix, row sum-to-one; A C G U
#     # assert len(s) > 1, "the length of rna should be at least 2!"
#     n = s.dim()[0][0]
#     counts = defaultdict(lambda: defaultdict(lambda: dy.scalarInput(float('-inf')))) # number of structures
#     for k in range(n):
#         counts[k][k]= dy.scalarInput(0.)
#         counts[k][k-1] = dy.scalarInput(0.)
#     for j in range(1, n):
#         for i in range(0, j):
#             counts[i][j] = dy.logsumexp([counts[i][j], counts[i][j-1]+(-unpair_score/kT)]) #torch.logaddexp(counts[i][j], counts[i][j-1]+(-unpair_score/kT)) # x.
#             if j-i>sharpturn:
#                 score_ij = dy.scalarInput(float('-inf')) #torch.tensor(-torch.inf, device=device)
#                 prob_sum = dy.scalarInput(1e-32) #prob_sum = 0
#                 for il, jr in pairs_index: 
#                     prob = s[i][il]*s[j][jr] + dy.scalarInput(1e-32)
#                     # prob_sum += prob
#                     # if prob > 0:
#                     pair_ij = nucs[il]+nucs[jr]
#                     score_ij = dy.logsumexp([score_ij, dy.log(prob)+(-pair2score[pair_ij]/kT)]) #torch.logaddexp(score_ij, torch.log(prob)+(-pair2score[pair_ij]/kT))
#                 # if prob_sum > 0:
#                 counts_right = counts[i+1][j-1] 
#                 for t in range(0, i):
#                     counts_left = counts[t][i-1] 
#                     counts[t][j] = dy.logsumexp([counts[t][j], counts_left+counts_right+score_ij]) # torch.logaddexp(counts[t][j], counts_left+counts_right+score_ij) # x(x)
#                 counts[i][j] = dy.logsumexp([counts[i][j], counts_right+score_ij]) #torch.logaddexp(counts[i][j], counts_right+score_ij) # (x)
#     return counts


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
    inside = inside_forward_stoc_log_dy(ts_input, device)
    # set records
    log = [] if log_last is None else log_last
    time_fw = 0.
    time_bw = 0.
    for epoch in range(num_step):
        start_fw = time.time()
        x.set(0.)
        count = -inside[0][l-1]
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
    print('partition:', inside_forward_pt(seq)[0][l-1])
    from partition_stoc import inside_forward_stoc_pt
    pt_np = inside_forward_stoc_pt(ts)
    print('pt_np:', pt_np[0][l-1])
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
    print('partition:', inside_forward_pt(seq)[0][l-1])
    return ts, log, time_fw, time_bw


def energy_score(seq, ss):
    pairs_list = extract_pairs_list(ss)
    # print('pairs_list: ', pairs_list)
    n_unpair = len(ss) - len(pairs_list)*2
    # print(f'n_unpair: {n_unpair}, n_pair: {len(pairs_list)}')
    score = n_unpair*unpair_score/kT
    for pair in pairs_list:
        i = pair[0]
        j = pair[1]
        pair_ij = seq[i]+seq[j]
        assert j-i > sharpturn
        if j-i > sharpturn and pair_ij in pair2score:
            score_ij = pair2score[pair_ij]/kT
            score += score_ij
        else:
            score += 2*unpair_score/kT
    return score


def energy_score_dy(seq, ss, device='cpu'):
    from partition_stoc import unpair_index, mispair_score
    pairs_list = extract_pairs_list(ss)
    print('pairs_list: ', pairs_list)
    n_unpair = len(ss) - len(pairs_list)*2
    print(f'n_unpair: {n_unpair}, n_pair: {len(pairs_list)}')
    score = dy.scalarInput(n_unpair*unpair_score/kT)
    score_mispair = dy.scalarInput(0.)
    for pair in pairs_list:
        i = pair[0]
        j = pair[1]
        assert j-i > sharpturn
        for il, jr in pairs_index:
            pair_ij = nucs[il]+nucs[jr]
            score_ij = pair2score[pair_ij]/kT
            prob = seq[i][il]*seq[j][jr]
            score += prob*score_ij
        for il, jr in unpair_index:
            score_mispair_ij = mispair_score/kT*2
            prob_mispair = seq[i][il]*seq[j][jr]
            score_mispair += prob_mispair*score_mispair_ij
    return score, score_mispair


def min_score(ss, lr, num_step, init=None, log_last=None, device='cpu'):
    pairs_list = extract_pairs_list(ss)
    print('pairs_list: ', pairs_list)
    n_unpair = len(ss) - len(pairs_list)*2
    m = dy.ParameterCollection()
    l = len(ss)
    ts = m.add_parameters((l, 4))
    if init is None:
        ts.set_value(rand_arr(l))
    else:
        ts.set_value(init)
    # build computational graph
    dy.renew_cg()
    x = dy.scalarInput(0.)
    ts_input = ts + x
    score, score_mispair = energy_score_dy(ts_input, ss, device)
    # set records
    log = [] if log_last is None else log_last
    time_fw = 0.
    time_bw = 0.
    for epoch in range(num_step):
        start_fw = time.time()
        x.set(0.)
        print(f'step: {epoch: 4d}, score: {score.value()}', f'score_mispair: {score_mispair.value()}')
        end_fw= time.time()
        time_fw += end_fw - start_fw
        start_bw = time.time()
        score.backward()
        # score_mispair.backward()
        end_bw = time.time()
        time_bw += end_bw - start_bw
        # print(ts.grad)
        ts_next = ts.value() - lr*ts.gradient()
        ts_next = projection_simplex_np_batch(ts_next)
        log.append(score.value())
        norm_diff = np.linalg.norm(ts.value()-ts_next)/l
        if norm_diff < 1e-6:
            ts.set_value(ts_next)
            break
        ts.set_value(ts_next)
    print('optimal solution:')
    ts = ts.value()
    for i in range(l):
        print(f"{i:3d}, A: {ts[i][0]:.4f}, C: {ts[i][1]:.4f}, G: {ts[i][2]:.4f}, U: {ts[i][3]:.4f}")
    print('optimal sequence:')
    seq = "".join([nucs[index] for index in np.argmax(ts, axis=1)])
    print(seq)
    print(ss)
    print('violated pairs:')
    for pair in pairs_list:
        i, j = pair
        p = seq[i]+seq[j]
        if p not in pair2score:
            print(i, j, p)
    print(f'n_unpair: {n_unpair}, n_pair: {len(pairs_list)}')
    print(f'optimal: {n_unpair - len(pairs_list)*3}')
    print(f'diff: {norm_diff:.8e}')
    score = energy_score(seq, ss)
    print(f'score: {score}')
    # print('partition:', inside_forward_pt(seq)[0][l-1])
    return ts, log, time_fw, time_bw


def max_prob(ss, lr, num_step, ts_init=None, log_last=None, device='cpu'):
    from partition_stoc import inside_forward_stoc_pt, seq2arr
    from rw_simple import pairs_match, init
    pairs_pos = pairs_match(ss)
    seq_init = init(ss, pairs_pos)
    ts_init = seq2arr(seq_init)
    print(f'seq_init: {seq_init}')
    pairs_list = extract_pairs_list(ss)
    print('pairs_list: ', pairs_list)
    n_unpair = len(ss) - len(pairs_list)*2
    # build computational graph
    dy.renew_cg()
    # dy.renew_cg(immediate_compute = True, check_validity = True)
    m = dy.ParameterCollection()
    l = len(ss)
    ts = m.add_parameters((l, 4))
    # print('parameters_list:')
    # print(m.parameters_list())
    if ts_init is None:
        ts.set_value(rand_arr(l))
    else:
        ts.set_value(ts_init)
    x = dy.scalarInput(0.)
    ts_input = ts + x
    score, score_mispair = energy_score_dy(ts_input, ss, device)
    inside = inside_forward_pt_stoc_log_dy(ts_input, device)
    prob_log = score + inside[0][l-1]
    # set records
    log = [] if log_last is None else log_last
    time_fw = 0.
    time_bw = 0.
    for epoch in range(num_step):
        # pt_np = inside_forward_stoc_pt(ts_input.value())
        start_fw = time.time()
        x.set(0.)
        print(f'step: {epoch: 4d}, prob: {np.exp(-prob_log.value()):.4e}, score: {score.value():.4e}, pt: {inside[0][l-1].value():.4e}, score_mispair: {score_mispair.value():.4e}')
        end_fw= time.time()
        time_fw += end_fw - start_fw
        start_bw = time.time()
        prob_log.backward()
        # score_mispair.backward()
        end_bw = time.time()
        time_bw += end_bw - start_bw
        # print(ts.grad)
        ts_next = ts.value() - lr*ts.gradient()
        ts_next = projection_simplex_np_batch(ts_next)
        log.append(prob_log.value())
        norm_diff = np.linalg.norm(ts.value()-ts_next)/l
        if norm_diff < 1e-9:
            ts.set_value(ts_next)
            break
        ts.set_value(ts_next)
    print('optimal solution:')
    ts = ts.value()
    for i in range(l):
        print(f"{i:2d}, A: {ts[i][0]:.4f}, C: {ts[i][1]:.4f}, G: {ts[i][2]:.4f}, U: {ts[i][3]:.4f}")
    print('optimal sequence:')
    seq = "".join([nucs[index] for index in np.argmax(ts, axis=1)])
    print(seq)
    print(ss)
    print('pairs:', pairs_list)
    print('violated pairs:')
    for pair in pairs_list:
        i, j = pair
        p = seq[i]+seq[j]
        if p not in pair2score:
            print(i, j, p)
    print(f'n_unpair: {n_unpair}, n_pair: {len(pairs_list)}')
    print(f'optimal: {n_unpair - len(pairs_list)*3}')
    print(f'diff: {norm_diff:.8e}')
    pt = inside_forward_pt(seq)[0][l-1]
    print('partition:', pt)
    pt_np = inside_forward_stoc_pt(ts)
    print('pt_np:', pt_np[0][l-1])
    score = energy_score(seq, ss)
    print('score:', score)
    from rw_simple import prob as prob_simple
    try:
        prob_hard = prob_simple(seq, ss)
    except:
        prob_hard = 0.
    print(f'prob: {prob_hard:.4e}' )
    np.save('ts.npy', ts)
    return ts, log, time_fw, time_bw, prob_hard



def run_batch(f, l_max, lr, num_step, device, folder="data/partition_dy"):
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
        d['seq'] = seq.tolist() # .cpu().detach().numpy().tolist()
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
            
            
def eterna(lr=0.5, num_step=500):
    path = 'data/eterna100/rna_inverse_v249_2.csv'
    df = pd.read_csv(path)
    data = []
    for i, row in df.iterrows():
        # if i < 4:
        #     continue
        target = row['structure']
        print(i, row['puzzle_name'])
        print('target: ', target)
        start_time = time.time()
        # max_prob(target, lr, num_step)
        ts, log, time_fw, time_bw, prob_hard = max_prob(target, lr, num_step)
        finish_time = time.time()
        
        data.append([row['puzzle_name'], target, np.array2string(ts, separator=','), log[-1], finish_time-start_time, time_fw, time_bw, log, prob_hard])
        filename = f"sgd_simple_step{num_step}_lr{lr}.csv"
        # if i == 2:
        #     break
        if (i+1)%10 == 0 or i==1:
            df_arw = pd.DataFrame(data, columns=('puzzle_name', 'structure', 'rna', 'objective', 'time', 'time_fw', 'time_bw','log', 'prob_hard'))
            df_arw.to_csv(filename)
    df_arw = pd.DataFrame(data, columns=('puzzle_name', 'structure', 'rna', 'objective', 'time', 'time_fw', 'time_bw','log', 'prob_hard'))
    df_arw.to_csv(filename)
        

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
    parser.add_argument("--step", type=int, default=400)
    parser.add_argument("--device", '-d', type=str, default="cpu")
    parser.add_argument("--sharpturn", type=int, default=3)
    parser.add_argument("--min", action='store_true')
    parser.add_argument("--batch", '-b', action='store_true')
    parser.add_argument("--eterna", '-e', action='store_true')
    args = parser.parse_args()
    print('args:')
    print(args)
    set_sharpturn(args.sharpturn)
    
    l = args.length
    lr = args.lr
    num_step = args.step
    device = args.device
    
    if args.eterna:
        eterna(lr, num_step=num_step)
        exit(0)
    
    # ss_target = "(((((.....))..((.........)))))"
    ss_target = '((((((.....(((....(((...(((((.............)))))....)))..)))..))))))'
    # ss_target = '(((((((....(((...........)))((((((((..(((((((((((((((((((...(((((......))))).)))))).)))))))))))))..))))))))..)))))))'
    path = 'data/eterna100/rna_inverse_v249_2.csv'
    df = pd.read_csv(path)
    # ss_target = df['structure'][1]
    if args.batch:
        run_batch(f, l, lr, num_step, device)
    else:
        # min_score(ss_target, lr, num_step)  
        max_prob(df['structure'][4], lr, num_step)
        # dy.renew_cg(immediate_compute = False, check_validity = True)
        # max_prob(df['structure'][5], lr, num_step)
    