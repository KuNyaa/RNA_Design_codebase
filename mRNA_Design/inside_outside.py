from collections import defaultdict
import sys

#match = set(["CG", "GC", "AU", "UA"]) #, "GU", "UG"])
from nussinov import match, sharpturn, Float
import numpy as np

class Viterbi(float): # (max, +) semiring
    def __iadd__(self, other):
        self = Viterbi(max(self, other)) # must return Float not float
        return self

def inside_forward(x):
    n = len(x)
    x = " " + x # so that we use 1-based index (NOT b/w nuc indices)
    Q = defaultdict(lambda: defaultdict(float))
    for j in range(1, n+1):
        Q[j-1][j] = 1
    for j in range(1, n+1):
        for i in Q[j-1]:
            Q[j][i] += Q[j-1][i] #* np.exp(-1)
            if x[i-1]+x[j] in match and j-i >= sharpturn:
                for k in Q[i-2]:
                    Q[j][k] += Q[i-2][k] * Q[j-1][i] * np.exp(match[x[i-1]+x[j]])
        #print(j, Q[j])
    #print()
    return np.log(Q[n][1])

def inside_viterbi(x):
    n = len(x)
    x = " " + x # so that we use 1-based index (NOT b/w nuc indices)
    Q = defaultdict(lambda: defaultdict(lambda : Viterbi(-np.inf)))
    for j in range(1, n+1):
        Q[j-1][j] = 0
    for j in range(1, n+1):
        for i in Q[j-1]:
            Q[j][i] += Q[j-1][i] # max=
            if x[i-1]+x[j] in match and j-i >= sharpturn:
                for k in Q[i-2]:
                    Q[j][k] += Q[i-2][k] + Q[j-1][i] + match[x[i-1]+x[j]] # max=
        #print(j, Q[j])
    #print()
    return Q[n][1]

def inside_forward_log(x):
    n = len(x)
    x = " " + x # so that we use 1-based index (NOT b/w nuc indices)
    Q = defaultdict(lambda: defaultdict(lambda: Float(-np.inf)))
    for j in range(1, n+1):
        Q[j-1][j] = Float(0)
    for j in range(1, n+1):
        for i in Q[j-1]:
            Q[j][i] += Q[j-1][i] #- 1
            if x[i-1]+x[j] in match and j-i >= sharpturn:
                for k in Q[i-2]:
                    Q[j][k] += Q[i-2][k] + Q[j-1][i] + match[x[i-1]+x[j]]
        #print(j, Q[j])
    #print()
    return Q[n][1]

def outside_forward(x, Q): # Q: inside
    n = len(x)
    x = " " + x # so that we use 1-based index (NOT b/w nuc indices)
    O = defaultdict(lambda: defaultdict(int)) # outside
    p = defaultdict(lambda: defaultdict(int)) # pairing
    O[n][1] = 1
    for j in range(n, 0, -1):
        for i in Q[j-1]:
            O[j-1][i] += O[j][i]
            if x[i-1]+x[j] in match and j-i >= sharpturn:
                for k in Q[i-2]:  # from out(k,j) to two children
                    O[i-2][k] += O[j][k] * Q[j-1][i]
                    O[j-1][i] += O[j][k] * Q[i-2][k]
                    p[j][i-1] += O[j][k] * Q[i-2][k] * Q[j-1][i]
        print(j, O[j])
    for j in p:
        for i in p[j]:
            print(i, j, p[j][i])
    return O, p

if __name__ == "__main__":
    #x = sys.argv[1] if len(sys.argv) >= 2 else "CCAAAGG"
    #Q = inside_forward(x)
    #O, p = outside_forward(x, Q)
    totalQ = Float(-np.inf)    
    logQs = []
    for i, line in enumerate(sys.stdin, 1):
        line = line.strip()
        logQ = inside_forward(line) # log-space is ~2x slower than exp-space! logaddexp VERY SLOW => logsumexp reduce
        best = inside_viterbi(line)
        print("%.4f %d %s" % (logQ, best, line))
        totalQ += logQ
        logQs.append((logQ, best))
        if i % 1000 == 0:
            print(i, "...", file=sys.stderr)
    avgQ = Float(-np.inf) # not simple average of logQs! (log-space average)
    logm = np.log(len(logQs))
    for logQ, viterbi in logQs:
        avgQ += logQ - logm # log(1/m)
    print("totalQ: %.5f, avgQ: %.5f, max: %s, min: %s, mfe: %d" % (totalQ, avgQ, max(logQs), min(logQs), max(logQs, key=lambda x: x[1])[1]))