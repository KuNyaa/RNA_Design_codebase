import numpy as np

import heapq

pairs = {'au', 'gc', 'gu'}


def match(x, y):
    return (x+y).lower() in pairs or (y+x).lower() in pairs


def hash_grid(row, col, n):
    return n*row + col


class MinHeap:
 
    def __init__(self, k):
        self.pool = []
        self.k = k
 
    def add(self, item):
 
        # if the min-heap's size is less than `k`, push directly
        if len(self.pool) < self.k:
            heapq.heappush(self.pool, item)
            return True
 
        # otherwise, compare to decide push or not
        elif self.pool[0].score < item.score:
            heapq.heappushpop(self.pool, item)
            return True
        return False
    
    def pop(self):
        return heapq.heappop(self.pool)
    
    
class Cand:
    
    def __init__(self, start, end, left, right, score, label, score_left, score_right, idx):
        self.start = start
        self.end = end
        self.left = left
        self.right = right
        self.score = score
        self.label = label
        self.score_left = score_left
        self.score_right = score_right
        self.idx = idx
        assert len(label) == self.end-self.start+1, f"label: {self.label}, start: {self.start}, end: {self.end}"
    
    def __gt__(self, other):
        return self.score < other.score
    
    def __lt__(self, other):
        return self.score > other.score
    
    def __ge__(self, other):
        return self.score <= other.score
    
    def __le__(self, other):
        return self.score >= other.score
    
    def __eq__(self, other):
        return self.score == other.score
    
    def __str__(self):
        return f"{self.score}, \'{self.label}\'"
    
    def __repr__(self):
        return f"{self.score}, \'{self.label}\'"


class HyperArc:
    
    def __init__(self, start, end, left, right, split, opt, k):
        self.start = start
        self.end = end
        self.left = left
        self.right = right
        self.split = split
        self.k = k
        self.opt = opt
        # self.seq_lefts = seq_lefts
        # self.seq_rights = seq_rights
        self.bitset = set()  # indicate the existence of row*k+col
        self.mincand = MinHeap(k*3)
        

class HyperArcList:
    
    def __init__(self, hyperarc_list, k):
        self.k = k
        self.hyperarc_list = hyperarc_list
        self.mincand = MinHeap(k*2+len(self.hyperarc_list))
        
    def initilize(self):
        mincand = self.mincand
        for i, harc in enumerate(self.hyperarc_list):
            bitset = harc.bitset
            k = harc.k
            cand_first = Cand(harc.start, harc.end, harc.left, harc.right, score=harc.opt[0], label=harc.opt[1], score_left=harc.opt[2], score_right=harc.opt[3], idx=i)
            mincand.add(cand_first)
            bitset.add(hash_grid(harc.left, harc.right, k))
        
    def extract_best(self, bests):
        if self.mincand.pool:
            # pop out the best
            cand_best = self.mincand.pop()
            return cand_best
        else:
            return None


def algo_3(s, k=10):
    assert len(s) > 1, "the length of rna should be at least 2!"
    n = len(s)
    bests = {}
    harcs = {}
    counts = np.ones((n, n)) # number of structures
    opts = np.zeros((n, n) ,dtype=object)
    for row in range(n):
        opts[row, row] = (0, '.', 0, 0) # tuple: (num_pair, prediction, num_left, num_right)
        key = hash_grid(row, row, n)
        if key not in bests:
            bests[key] = [Cand(start=row, end=row, left=0, right=0, score=0, label='.', score_left=None, score_right=None, idx=None)]
    for l in range(1, n):
        for i in range(0, n-l):
            j = i + l
            key = hash_grid(i, j, n)
            if key not in bests:
                bests[key] = []
            # Case 1 doesn't exist here because l starts from 1
            # Case 2
            counts[i, j] = counts[i, j-1]    
            opt_2 = opts[i, j-1][0], opts[i, j-1][1]+".", opts[i, j-1][0], 0
            opts[i, j] = opt_2
            harc = HyperArc(start=i, end=j, left=1, right=0, split=j, opt=opt_2, k=k)
            harc_list = []
            harc_list.append(harc)
            # Case 3
            for t in range(i, j):
                if match(s[t],  s[j]):
                        
                    # deal with number of structures
                    count_left = counts[i, t-1] if i<(t-1) else 1
                    count_right = counts[t+1, j-1] if (t+1)<(j-1) else 1
                    counts[i, j] += count_left*count_right
                    
                    # deal with best k
                    # structure on the left of t
                    if i<=(t-1):
                        seq_left = opts[i, t-1]
                        left = 1
                    else:
                        seq_left = (0, "")
                        left = 0

                    # structure on the right of t
                    if (t+1)<=(j-1):
                        seq_right = opts[t+1, j-1]
                        right = 1
                    else:
                        seq_right = (0, "")
                        right = 0
                    opt_1 = seq_left[0] + seq_right[0] + 1, seq_left[1] + "(" + seq_right[1] + ")", seq_left[0], seq_right[0]+1
                    harc = HyperArc(start=i, end=j, left=left, right=right, split=t, opt=opt_1, k=k)
                    harc_list.append(harc)
                    if opt_1[0] > opts[i, j][0]:
                        opts[i, j] = opt_1
            hyperarc_list = HyperArcList(harc_list, k)
            hyperarc_list.initilize()
            if key not in harcs:
                harcs[key] = hyperarc_list
    lazy_kth_best(0, n-1, n, k, k, bests, harcs)
    best_1 = opts[0, -1][0], opts[0, -1][1]
    num_struct = int(counts[0, -1]) 
    best_k = [(cand.score, cand.label) for cand in bests[hash_grid(0, n-1, n)]]
    return best_1, num_struct, best_k
    # return opts[0, -1], int(counts[0, -1]), bests[hash_grid(0, n-1, n)]


def lazy_kth_best(start, end, n, kth, k, bests, harcs):
    if start < end:
        key = hash_grid(start, end, n)
        while len(bests[key]) < kth:
            if bests[key]:
                best_last = bests[key][-1]
                lazy_next(harcs[key], best_last, bests, harcs, n)
            if harcs[key].mincand.pool:
                bests[key].append(harcs[key].extract_best(bests))
            else:
                break
        return


def lazy_next(harc_list, cand_best, bests, harcs, n):
    mincand = harc_list.mincand
    idx_harc = cand_best.idx
    harc = harc_list.hyperarc_list[idx_harc]
    k = harc.k
    bitset = harc.bitset
    #axis left
    start_left = harc.start
    end_left = harc.split - 1
    if cand_best.left != 0 and start_left < end_left and cand_best.left and cand_best.left+1 <= k:
        left = cand_best.left + 1
        lazy_kth_best(start_left, end_left, n, left, k, bests, harcs)  # recursive call
        bests_left = bests[hash_grid(start_left, end_left, n)]
        key = hash_grid(left, cand_best.right, k)
        if len(bests_left)>=left and key not in bitset:
            seq_left = bests_left[left-1].label
            score_left = bests_left[left-1].score
            score_right = cand_best.score_right
            score_new = score_left + score_right
            label_new = seq_left + cand_best.label[harc.split-harc.start:]
            assert len(label_new)==harc.end-harc.start+1, f"len(label_new): {len(label_new)}, length: {harc.end-harc.start+1}, start: {harc.start}, end: {harc.end}, label_last: {cand_best.label}, label_new: {label_new}"
            cand_left = Cand(harc.start, harc.end, left, cand_best.right, score=score_new, label=label_new, score_left=score_left, score_right=score_right, idx=cand_best.idx)
            mincand.add(cand_left)
            bitset.add(key)

    #axis right
    start_right = harc.split + 1
    end_right = harc.end - 1
    if cand_best.right != 0 and start_right < end_right and cand_best.right and cand_best.right+1 <= k:
        right = cand_best.right + 1
        lazy_kth_best(start_right, end_right, n, right, k, bests, harcs)  # recursive call
        bests_right = bests[hash_grid(start_right, end_right, n)]
        key = hash_grid(cand_best.left, right, k)
        if len(bests_right)>=right and key not in bitset:
            seq_right = bests_right[right-1].label
            # print(f"start_right: {start_right}, end_right: {end_right}")
            score_left = cand_best.score_left
            score_right = bests_right[right-1].score + 1
            score_new = score_left + score_right
            label_new = cand_best.label[:harc.split-harc.start] + "(" + seq_right + ")"
            assert len(label_new)==harc.end-harc.start+1, f"len(label_new): {len(label_new)}, length: {harc.end-harc.start+1}, start: {harc.start}, end: {harc.end}, label_last: {cand_best.label}, label_new: {label_new}"
            cand_right = Cand(harc.start, harc.end, cand_best.left, right, score=score_new, label=label_new, score_left=score_left, score_right=score_right, idx=cand_best.idx)
            mincand.add(cand_right)
            bitset.add(key)