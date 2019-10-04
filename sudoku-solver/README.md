# Sudoku Solver

A solution to [http://www.cs.toronto.edu/~lyan/csc384/a2/A2Sudoku.pdf](http://www.cs.toronto.edu/~lyan/csc384/a2/A2Sudoku.pdf)

## How to Run

```bash
python3 SudokuSolver.py puzzle_filename algorithm
```

where `puzzle_filename` is a `.txt` file containing an incomplete sudoku to solve and `algorithm` should be one of the following:

- BF: Brute Force
- BT: Back Tracking
- FC-MRV: Forward Checking with Minimum Remaining Values

This python script presents three algorithms for finding a solution of an incomplete sudoku.

## Brute Force

The simplest algorithm is Brute Force Searching. In this algorithm, the programs just iterates through all the grids in the sudoku puzzle and fills in a number if empty. When duplicate value is found in one of the rows, columns or 3x3 squares, the program clears all the numbers and continue searching.

This method can take a while and is NP-complete. Its time complexity could be $O(9^n)$ where $n$ is the number of empty grids.

## Back Tracking

The sudoku problem can be regarded as a CSP (Constraint Satisfaction Problem). 

So we can apply back-tracking algorithm. This method selects an empty grid at a time and fills in a consistent value. Then it recursively selects next empty grid and fills in a value again. When it finds a solution, it returns the solution. If the sudoku is still unfinished but the program could not fill in an empty grid because every value conflicts, the program backtracks to last step, clear the assignment and tries another value to see if it could reach a solution.

```python
def back_tracking(sudoku):
    variable = find_unassigned(sudoku)
    if variable is None:
        return sudoku
    for value in range(1, 10):
        if check(sudoku, variable, value):
            sudoku[variable] = value
            result = back_tracking(sudoku)
            if result:
                return result
        sudoku[variable] = 0

    return False
```

## Forward Checking with Minimum Remaining Values Heuristics

To further narrow down the search space, we could do forward checking before we continue backtracking.
Whenever we assign a variable, we delete inconsistent values from unassigned variables related to the just assigned variable. After the process, if we find an unassigned variable with no legal values remained, we could assert that the current assignment is impossible for a solution. In this way, we could prune invalid assignment earlier and narrow down the search space.

What's more, we could further narrow down search space by choosing the variable with minimum remaining values so we could try less values. To do so, we need to generate all legal values for every unassigned variables and sort the variables by their number of legal values in an ascending order.

```python
def forward_checking(sudoku):
    data = find_unassigned_mrv(sudoku)
    if data is None:
        return sudoku
    variables, values = data
    variable = variables[0]
    # We have already ensured the values are consistent
    for value in values[variable]:
        sudoku[variable] = value
        viable = True

        # Forward checking
        # Remove all inconsistent values
        for neighbour in connected[variable]:
            if values.get(neighbour) is None:
                continue
            if value in values[neighbour] and len(values[neighbour]) == 1:
                viable = False
                break

        if viable:
            result = forward_checking(sudoku)
            if result:
                return result

        sudoku[variable] = 0
    return False
```

## Results

The results were retrieved on a computer with Intel i7-8700K @ 4.30 GHz

Total Clock Time (in ns):

|Puzzle|Brute Force|Back Tracking|Forward Checking with MRV|
|------|-----------|-------------|-------------------------|
|1|2019400|1969900|6951900|
|2|12965000|15981500|9943600|
|3|22912000|29911400|28898400|
|4|8976000|12997000|14986300|
|5|4962300|6980900|13956700|

Search Clock Time (in ns):

|Puzzle|Brute Force|Back Tracking|Forward Checking with MRV|
|------|-----------|-------------|-------------------------|
|1|970400|1994500|5010600|
|2|10974400|14960900|9946700|
|3|20944700|29919700|27929700|
|4|6981000|10971600|14984400|
|5|3993800|4014400|13962300|

Number of nodes generated:

|Puzzle|Brute Force|Back Tracking|Forward Checking with MRV|
|------|-----------|-------------|-------------------------|
|1|139|95|42|
|2|1246|912|57|
|3|2303|1620|158|
|4|614|543|83|
|5|341|292|67|

It seems that the FC-MRV algorithm does not outperform other algorithms in terms of clock time. However, it expands much less nodes than other algorithms. This means that much time is cost in finding all legal values for unassigned variables.