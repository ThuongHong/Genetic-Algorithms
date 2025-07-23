# N-Queens

The N-Queens problem requires placing N queens on an N×N chessboard such that no two queens attack each other. In this experiment, we approach the problem using three algorithms:

# Genetic Algorithm
- Each **individual** represents one full-board solution (a permutation of size N).
- **Fitness function**: Number of non-attacking queen pairs. The maximum fitness is $f_{max} = \binom{N}{2}$.
- **Mutation**: Swap two random positions in the chromosome.
- **Mate**: Randomized crossover with three possibilities:
    + 25%: mutate based on the first parent
    + 25%: mutate based on the second parent
    + 50%: generate a completely new chromosome

# Result

We tested all three algorithms on board sizes from **N = 4 to N = 40**, with a timeout threshold of **30 seconds** for each run.

- **Genetic Algorithm** performs robustly across all test cases. It has an **average runtime of 0.47s**, with the **worst-case at 3.26s**.
- **Backtracking** is effective for small N, but fails to find a solution within the timeout at **N = 28**.
- **Brute Force** becomes infeasible early, failing already at **N = 14**, due to factorial growth in the number of permutations.

This demonstrates that while **Backtracking** is a decent choice for small sizes, and **Brute Force** is purely educational, the **Genetic Algorithm** offers a scalable solution that works even for large N.

