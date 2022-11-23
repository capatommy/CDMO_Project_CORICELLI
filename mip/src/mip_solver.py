from pulp import *
import datetime
import os
from itertools import combinations
import numpy as np
import utils
import time
from tqdm import tqdm
import cplex

class SolverMIP:

    def __init__(self, timeout=300, rotation=False):
        self.timeout = timeout
        self.rotation = rotation

    def solve(self, data):

        self.model = LpProblem("VLSI", LpMinimize)

        self.plate_width, self.circuits = data
        self.circuits_num = len(self.circuits)
        self.w, self.h = ([ i for i, _ in self.circuits ], [ j for _, j in self.circuits ])        

        upper_bound = sum(self.h)
        lower_bound = max(self.h)
        CIRCUITS = range(self.circuits_num)

        l = LpVariable("l", lowBound=lower_bound, upBound=upper_bound, cat = LpInteger)
        self.model += l
        
        x = [
            LpVariable(
                f"x_{i}", lowBound=0, upBound=int(self.plate_width - self.w[i]), cat=LpInteger
            ) for i in CIRCUITS
        ]

        y = [
            LpVariable(
                f"y_{i}", lowBound=0, upBound=int(upper_bound - self.h[i]), cat=LpInteger
            ) for i in CIRCUITS
        ]
        
        for i in CIRCUITS:
            self.model += y[i] + self.h[i] <= l, f"Y-axis of {i}-th coordinate bound"

        # Booleans for OR condition
        D = range(2)
        delta = LpVariable.dicts(
            "delta",
            indices=(CIRCUITS, CIRCUITS, D),
            cat=LpBinary,
            lowBound=0,
            upBound=1,
        )

        max_circuit = np.argmax(np.asarray(self.w) * np.asarray(self.h))
        self.model += x[max_circuit] == 0, "Max circuit in x-0"
        self.model += y[max_circuit] == 0, "Max circuit in y-0"

        # Non-Overlap constraints, at least one needs to be satisfied
        for i in CIRCUITS:
            for j in CIRCUITS:
                if i < j:
                    if self.w[i] + self.w[j] > self.plate_width:
                        self.model += delta[i][j][0] == 1
                        self.model += delta[j][i][0] == 1

                    self.model += x[i] + self.w[i] <= x[j] + (delta[i][j][0]) * self.plate_width
                    self.model += x[j] + self.w[j] <= x[i] + (delta[j][i][0]) * self.plate_width
                    self.model += y[i] + self.h[i] <= y[j] + (delta[i][j][1]) * upper_bound
                    self.model += y[j] + self.h[j] <= y[i] + (delta[j][i][1]) * upper_bound
                    self.model += (
                        delta[i][j][0] + delta[j][i][0] + delta[i][j][1] + delta[j][i][1]
                        <= 3
                    )
        try:
            self.model.solve(CPLEX_PY(mip=True, msg=False, timeLimit=self.timeout))
        except BaseException as err:
            logging.error(f"Unexpected {err}")
            print("Status: ", LpStatus[self.model.status])
            return None, 0
        
        if self.model.solutionTime > self.timeout:
            print("Timeout reached, no optimal solution provided")
            return None, 0
        else:
            height_sol = round(value(self.model.objective))
            print("HEIGHT: {}".format(height_sol))

            circuit_pos = []
            x_sol = []
            y_sol = []

            for v in self.model.variables():
                if str(v.name).startswith("x"):
                    x_sol.append(round(v.varValue))
                elif str(v.name).startswith("y"):
                    y_sol.append(round(v.varValue))

            for k in CIRCUITS:
                circuit_pos.append((self.w[k], self.h[k], x_sol[k], y_sol[k]))
            
            return ((self.plate_width, height_sol), circuit_pos), self.model.solutionTime
                

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