from operator import length_hint
import os
from z3 import *
from itertools import combinations
import numpy as np
import utils
import time
from tqdm import tqdm

class SolverSAT:

    def __init__(self, timeout=300, rotation=False):
        self.timeout = timeout
        self.rotation = rotation

    def amo(self, bool_vars):
        return [Not(And(pair[0], pair[1])) for pair in combinations(bool_vars, 2)]
    
    def alo(self, bool_vars):
        return Or(bool_vars)
    
    def eo(self, bool_vars):
        return self.amo(bool_vars) + [self.alo(bool_vars)]

    def flat(self, list_of_lists):
        return list(np.concatenate(list_of_lists).flat)

    def solve(self, data):
        self.plate_width, self.circuits = data
        self.circuits_num = len(self.circuits)
        self.w, self.h = ([ i for i, _ in self.circuits ], [ j for _, j in self.circuits ])

        upper_bound = sum(self.h)

        p = [[[Bool(f"p_{i}_{j}_{k}") for k in range(self.circuits_num)] for j in range(self.plate_width)] for i in range(upper_bound)]

        l = [Bool(f"l_{i}") for i in range(upper_bound)]

        r = None

        print("Building constraints...")

        no_overlapping = []
        for i in tqdm(range(upper_bound), desc= "NO OVERLAPPING", leave = False):
            for j in range(self.plate_width):
                no_overlapping += self.amo(p[i][j])

        eo_configurations = []

        if self.rotation:

            r = [Bool(f"r_{i}") for i in range(self.circuits_num)]

            for k in tqdm(range(self.circuits_num), "Exclusivity position in the board by each circuit", leave =False):
                all_configurations = []

                for y in range(upper_bound - self.h[k] + 1):
                    for x in range(self.plate_width - self.w[k] + 1):
                        
                        configurations = []
                        for oy in range(upper_bound):
                            for ox in range(self.plate_width):
                                if y <= oy < y + self.h[k] and x <= ox < x + self.w[k]:
                                    configurations.append(p[oy][ox][k])
                                else:
                                    configurations.append(Not(p[oy][ox][k]))

                        all_configurations.append(And(configurations))
                            
                no_rot_configurations = And(Not(r[k]), And(self.eo(all_configurations)))

                x_r_k , y_r_k = self.h[k], self.w[k]

                all_configurations = []

                for y in range(upper_bound - y_r_k + 1):
                    for x in range(self.plate_width - x_r_k + 1):
                        
                        configurations = []
                        
                        for oy in range(upper_bound):
                            for ox in range(self.plate_width):
                                if y <= oy < y + y_r_k and x <= ox < x + x_r_k:
                                    configurations.append(p[oy][ox][k])
                                else:
                                    configurations.append(Not(p[oy][ox][k]))

                        all_configurations.append(And(configurations))
                rot_configurations = And(r[k], And(self.eo(all_configurations)))

                eo_configurations += self.eo([no_rot_configurations, rot_configurations])
        else:
            for k in tqdm(range(self.circuits_num), "Exclusivity position in the board by each circuit", leave =False):
                all_configurations = []

                for y in range(upper_bound - self.h[k] + 1):
                    for x in range(self.plate_width - self.w[k] + 1):

                        configurations = []
                        for oy in range(upper_bound):
                            for ox in range(self.plate_width):
                                if y <= oy < y + self.h[k] and x <= ox < x + self.w[k]:
                                    configurations.append(p[oy][ox][k])
                                else:
                                    configurations.append(Not(p[oy][ox][k]))
                        all_configurations.append(And(configurations))
                            
                eo_configurations += self.eo(all_configurations)
        
        one_hot_length = self.eo([l[i] for i in tqdm(range(upper_bound), "One-hot encoding of the height of the board", leave=False)])

        length_circuits_positioning = [l[i] == And([Or(self.flat(p[i]))] + [Not(Or(self.flat(p[j]))) for j in range(i + 1, upper_bound)]) for i in tqdm(range(upper_bound), desc="Lenght consistency between circuits", leave=False)]

        max_h = np.argmax(self.h)
        highest_circuit_first = [And([p[i][j][k] if k == max_h 
            else Not(p[i][j][k]) for k in range(self.circuits_num) for j in range(self.w[max_h]) for i in tqdm(range(self.h[max_h]),desc="Set Highest circuit first", leave=False)])]

        solver = Solver()

        print("Adding constraints...")

        solver.add(no_overlapping)
        solver.add(eo_configurations)
        solver.add(one_hot_length)
        solver.add(length_circuits_positioning)
        solver.add(highest_circuit_first)

        solver.set("timeout", self.timeout*1000)

        print("Solving the task...")
        start_time = time.time()

        solution_found = False

        while True:
            if solver.check() == sat:

                model = solver.model()
                for k in range(upper_bound):
                    if model.evaluate(l[k]):
                        length_sol = k

            # prevent next model from using the same assignment as a previous model
                solver.add(self.alo([l[i] for i in range(length_sol)]))

                solution_found = True

            else:
            # break when it is impossible to improve anymore the length
                break
        
        if solution_found:
            length_sol += 1
            elapsed_time = time.time() - start_time

            circuit_pos = self.model_to_coordinates(model, p, self.plate_width, length_sol, self.circuits_num, r)

            return ((self.plate_width, length_sol), circuit_pos), elapsed_time

        elif solver.reason_unknown() == "timeout":
            print("Timeout reached, no optimal solution provided")
            return None, 0
        else:
            print("Unsatisfiable problem")
            return None, 0

    def model_to_coordinates(self, model, p, w, l, n, r=None):
        # Create solution array
        solution = np.array([[[is_true(model[p[i][j][k]]) for k in range(n)] for j in range(w)] for i in range(l)])
        circuits_pos = []

        for k in range(n):
            y_ids, x_ids = solution[:, :, k].nonzero()
            x = np.min(x_ids)
            y = np.min(y_ids)

            if r is None:
                circuits_pos.append((self.w[k], self.h[k], x, y))
            else:
                    if is_true(model[r[k]]):
                        circuits_pos.append((self.h[k], self.w[k], x, y))
                    else:
                        circuits_pos.append((self.w[k], self.h[k], x, y))

        return circuits_pos

if __name__ == "__main__":
    import sys
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    solver_dir = os.path.join(base_dir, "sat")
    sys.path.append(base_dir)
    from utils import solve_and_write, parse_arguments

    args = parse_arguments(main=False)
    solve_and_write(
        SolverSAT(timeout=args.timeout, rotation=args.rotation), 
        base_dir, solver_dir, ["res/instances/ins-{0}.txt".format(i) for i in range(41)], 
        average=args.average, stats=args.stats, plot=args.plot
    )

