import sys
import time

SUDOKU_SIZE = 9

connected = {}
expanded = 0


def generate_neighbours(sudoku):
    global connected
    for row in range(SUDOKU_SIZE):
        for col in range(SUDOKU_SIZE):
            if sudoku[row, col] == 0:
                connected[row, col] = set()
                for r in range(SUDOKU_SIZE):
                    if sudoku[r, col] == 0:
                        connected[row, col].add((r, col))
                for c in range(SUDOKU_SIZE):
                    if sudoku[row, c] == 0:
                        connected[row, col].add((row, c))
                _row = row - (row % 3)
                _col = col - (col % 3)
                for dr in range(3):
                    for dc in range(3):
                        if sudoku[_row + dr, _col + dc] == 0:
                            connected[row, col].add((_row + dr, _col + dc))
                connected[row, col].remove((row, col))


def find_unassigned(sudoku):
    for row in range(SUDOKU_SIZE):
        for col in range(SUDOKU_SIZE):
            if sudoku[row, col] == 0:
                return row, col
    return None


def find_unassigned_mrv(sudoku):
    unassigned = []
    for row in range(SUDOKU_SIZE):
        for col in range(SUDOKU_SIZE):
            if sudoku[row, col] == 0:
                unassigned.append((row, col))
    if not unassigned:
        return None
    values = {variable: viable_values(sudoku, variable) for variable in unassigned}
    unassigned.sort(key=lambda variable: len(values[variable]))
    return unassigned, values


def viable_values(sudoku, variable):
    values = set(range(1, SUDOKU_SIZE + 1))
    row, col = variable

    for r in range(SUDOKU_SIZE):
        num = sudoku[(r, col)]
        if num in values:
            values.remove(num)

    for c in range(SUDOKU_SIZE):
        num = sudoku[(row, c)]
        if num in values:
            values.remove(num)

    row -= (row % 3)
    col -= (col % 3)
    for dr in range(3):
        for dc in range(3):
            num = sudoku[(row + dr, col + dc)]
            if num in values:
                values.remove(num)

    return values


def brute_force(sudoku, row=0, col=0):
    global expanded
    expanded += 1
    if row >= SUDOKU_SIZE:
        return sudoku

    if sudoku[row, col] == 0:
        for value in range(1, SUDOKU_SIZE + 1):
            if check(sudoku, (row, col), value):
                sudoku[row, col] = value
                col += 1
                if col > 8:
                    row += 1
                    col = 0
                solution = brute_force(sudoku, row, col)
                if solution:
                    return solution
                col -= 1
                if col < 0:
                    row -= 1
                    col = 8
                sudoku[row, col] = 0
    else:
        col += 1
        if col > 8:
            row += 1
            col = 0
        return brute_force(sudoku, row, col)
    return False


def back_tracking(sudoku):
    global expanded
    expanded += 1
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


def forward_checking_wrapper(sudoku):
    generate_neighbours(sudoku)
    return forward_checking(sudoku)


def forward_checking(sudoku):
    global expanded
    expanded += 1
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


def usage():
    print('Usage: python SudokuSolver.py puzzle.txt algorithm(BF BT or FC-MRV)')


def check(sudoku, variable, value):
    return check_row(sudoku, variable[0], value) and \
           check_column(sudoku, variable[1], value) and \
           check_square(sudoku, variable[0], variable[1], value)


def check_square(sudoku, row, col, new_num):
    row -= (row % 3)
    col -= (col % 3)
    for dr in range(3):
        for dc in range(3):
            num = sudoku[(row + dr, col + dc)]
            if num == new_num:
                return False
    return True


def check_column(sudoku, col, new_num):
    for row in range(SUDOKU_SIZE):
        num = sudoku[(row, col)]
        if new_num == num:
            return False
    return True


def check_row(sudoku, row, new_num):
    for col in range(SUDOKU_SIZE):
        num = sudoku[(row, col)]
        if new_num == num:
            return False
    return True


def parse_sudoku(fn):
    sudoku = {}
    with open(fn, "r") as f:
        for row, line in enumerate(f):
            nums = [int(x) for x in line.split()]
            for col, num in enumerate(nums):
                sudoku[(row, col)] = num
    return sudoku


def print_sudoku(sudoku, f=sys.stdout):
    for row in range(SUDOKU_SIZE):
        for col in range(SUDOKU_SIZE):
            print(sudoku[row, col], end=' ', file=f)
        print('', file=f)


def print_performance(total_tik, search_tik, total_tok, search_tok, f=sys.stdout):
    print(f"Total clock time: {total_tok - total_tik} ns", file=f)
    print(f"Search clock time: {search_tok - search_tik} ns", file=f)
    print(f"Number of nodes generated: {expanded}", file=f)


def validate_solution(solution):
    # Row
    for row in range(SUDOKU_SIZE):
        nums = set()
        for c in range(SUDOKU_SIZE):
            num = solution[row, c]
            assert num not in nums, f"Duplicated number {num} found in row {row + 1}"
            nums.add(num)

    # Col
    for col in range(SUDOKU_SIZE):
        nums = set()
        for r in range(SUDOKU_SIZE):
            num = solution[r, col]
            assert num not in nums, f"Duplicated number {num} found in column {col + 1}"
            nums.add(num)

    # Square
    for s_row in range(3):
        for s_col in range(3):
            nums = set()
            row = s_row * 3
            col = s_col * 3
            for dr in range(3):
                for dc in range(3):
                    num = solution[row + dr, col + dc]
                    assert num not in nums, f"Duplicated number {num} found in square {s_row}, {s_col}"
                    nums.add(num)

    return True


ALGORITHMS = {'BF': brute_force, 'BT': back_tracking, 'FC-MRV': forward_checking_wrapper}

if __name__ == '__main__':
    total_tik = time.time_ns()
    if len(sys.argv) < 3:
        usage()
        sys.exit(1)
    puzzle_filename = sys.argv[1]
    algorithm = sys.argv[2]
    if algorithm not in ALGORITHMS:
        usage()
        sys.exit(1)
    sudoku = parse_sudoku(puzzle_filename)
    solution_f = open(puzzle_filename.replace("puzzle", "solution"), "w")
    performance_f = open(puzzle_filename.replace("puzzle", "performance"), "w")

    search_tik = time.time_ns()
    solution = ALGORITHMS[algorithm](sudoku)
    search_tok = time.time_ns()
    validate_solution(solution)
    if not solution:
        print("No solution found")
    else:
        print_sudoku(solution, solution_f)

    total_tok = time.time_ns()
    print_performance(total_tik, search_tik, total_tok, search_tok, performance_f)
    solution_f.close()
    performance_f.close()
