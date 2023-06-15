#!/usr/bin/env python3

import sys
import random

if __name__ == "__main__":

    AAs = []
    for line in open("coding_wheel.txt"):
        aa = line.split()[0]
        if aa != "STOP":
            AAs.append(aa)

    m = len(AAs)
    num = int(sys.argv[1])
    s = "Met " + " ".join(AAs[random.randint(0, m-1)] for _ in range(num-2)) + " STOP"
    print(s)
    print(s, file=sys.stderr)
