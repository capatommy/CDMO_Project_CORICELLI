from pulp import *
import datetime
import os
from itertools import combinations
import numpy as np
import utils
import time
from tqdm import tqdm

class SolverMIP:

    def __init__(self, timeout=300, rotation=False, solver_dir=None):
        self.timeout = timeout
        self.rotation = rotation
        self.solver_dir = os.path.join(solver_dir, "src")

    def solve(self, data):

        self.model = LpProblem("VLSI", LpMinimize)

        self.plate_width, self.circuits = data
        self.circuits_num = len(self.circuits)
        self.w, self.h = ([ i for i, _ in self.circuits ], [ j for _, j in self.circuits ])        

        upper_bound = sum(self.h)
        CIRCUITS = range(self.circuits_num)
        ROWS = range(self.plate_width)
        COLS = range(upper_bound) 

        cell = LpVariable.dicts("Cell", (CIRCUITS, ROWS, COLS), cat="Binary")

        for k in CIRCUITS:

            for r in ROWS:
                self.model += lpSum([cell[k][r][c] for c in COLS]) == 1

            for c in COLS:
                self.model += lpSum([cell[k][r][c] for r in ROWS]) == 1

        self.model.solve()

if __name__ == "__main__":
    import sys
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    solver_dir = os.path.join(base_dir, "mip")
    sys.path.append(base_dir)
    from utils import solve_and_write, parse_arguments

    args = parse_arguments(main=False)
    solve_and_write(
        SolverMIP(timeout=args.timeout, rotation=args.rotation, solver_dir=solver_dir), 
        base_dir, solver_dir, ["res/instances/ins-{0}.txt".format(i) for i in range(41)], 
        average=args.average, stats=args.stats, plot=args.plot
    )