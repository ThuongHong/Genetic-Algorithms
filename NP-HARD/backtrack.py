import time

def is_safe(row, col, board, N):
    for prev_row in range(row):
        if (board[prev_row] == col or
            abs(board[prev_row] - col) == abs(prev_row - row)):
            return False
    return True

def solve_bt(row, board, N):
    if row == N:
        return True
    for col in range(N):
        if is_safe(row, col, board, N):
            board[row] = col
            if solve_bt(row + 1, board, N):
                return True
    return False

def Backtracking(N):
    board = [-1] * N
    solve_bt(0, board, N)
    return tuple(board)