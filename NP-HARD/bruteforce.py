import time
from itertools import permutations

def is_valid_perm(perm):
    N = len(perm)
    for i in range(N):
        for j in range(i + 1, N):
            if abs(i - j) == abs(perm[i] - perm[j]):
                return False
    return True

def BruteForce(N):
    for perm in permutations(range(N)):
        if is_valid_perm(perm):
            return tuple(perm)
    return None