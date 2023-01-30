
from __future__ import print_function
import math, random, json
import numpy as np

# from pymoo.util.termination.collection import TerminationCollection
# from pymoo.util.termination.max_time import TimeBasedTermination

from pymoo.optimize import minimize
from pymoo.algorithms.nsga2 import nsga2
from pymop.problem import Problem
from pymoo.model.termination import Termination

def add_objects_nsga(scene_struct, mappings, shape_color_combos, args):

    # STEP 0 read the graph
    
    with open(args.graph_src_path, 'r') as f:
        constraints = json.load(f)['relationships']
    num_objects = len(constraints['front'])
    
    # STEP 1 Randomly select sizes, materials, rotation
    objects_all_info = []
    radiuses = []
    for _ in range(num_objects):
        size_name, r = random.choice(mappings['size'])
        mat_name, mat_name_out = random.choice(mappings['material'])
        rotation_theta = 360.0 * random.random()
        objects_all_info.append({
            'size':size_name,
            'material': mat_name,
            'material_out': mat_name_out,
            'rotation': rotation_theta,
        })
        radiuses.append(r)

    # STEP 2 TODO use NSGA to derive x, y positions, given the selected radius for each object

    # get constrints from args.
    tot_var = num_objects*2
    loBd = -3
    hiBd = 3
    
    class MyProblem(Problem):
        def __init__(self):
            # obj 1 = relations + margins
            # obj 2 = distances
            super().__init__(n_var=tot_var, n_obj=2, n_constr=0,
                            xl=loBd, xu=hiBd)

        def get_heuristics(self, all_points_at_gen):
            # x = [[x_a0, y_a0, x_a1, y_a1, ...],
            #     [x_a0, y_a0, x_a1, y_a1, ...],
            #     [x_a0, y_a0, x_a1, y_a1, ...]]

            all_heuristics = []
            for x in all_points_at_gen:
                # STEP 2.1 relations
                # NOTE combined with margin. ASSUMES that for each pair, a horizontal and a vertical constraint is defined
                totRel = 0
                for dir_name, dir_cons in constraints.items():
                    direction_vec = scene_struct['directions'][dir_name]
                    for i_tgt, sources in enumerate(dir_cons):
                        x_tgt, y_tgt = x[2*i_tgt], x[2*i_tgt+1]
                        for i_src in sources:
                            x_src, y_src = x[2*i_src], x[2*i_src+1]

                            dx, dy = x_src-x_tgt, y_src-y_tgt
                            assert direction_vec[2] == 0
                            margin = dx * direction_vec[0] + dy * direction_vec[1]
                            # print(margin)
                            # print(args.margin)
                            if margin < args.margin:
                                # if margin<0, then its in the opposite direction of where it is supposed to be
                                # if 0 < margin < args.margin, then margin is not respected
                                totRel += args.margin - margin

                # STEP 2.2 distances
                totDists, totMargins = 0, 0
                for i1 in range(num_objects):
                    x1, y1 = x[2*i1], x[2*i1+1]
                    r1 = radiuses[i1]
                    for i2 in range(i1+1, num_objects):
                        x2, y2 = x[2*i2], x[2*i2+1]
                        r2 = radiuses[i2]

                        dx, dy = x1-x2, y1-y2
                        dist_p2p = math.sqrt(dx * dx + dy * dy)
                        dist = dist_p2p - r1 - r2
                        if dist < args.min_dist:
                            totDists += args.min_dist-dist

                        # # STEP 2.3 margins
                        # for direction_name in ['left', 'right', 'front', 'behind']:
                        #     direction_vec = scene_struct['directions'][direction_name]
                        #     assert direction_vec[2] == 0
                        #     margin = dx * direction_vec[0] + dy * direction_vec[1]
                        #     if 0 < margin < args.margin:
                        #         totMargins += args.margin - margin

                all_heuristics.append([totRel, totDists]) # [totRel, totDists, totMargins]

            return np.array(all_heuristics)

        # Notes: x = [x_a0, y_a0, x_a1, y_a1, ...]
        def _evaluate(self, x, out, *args, **kwargs):
            heu_vals = self.get_heuristics(x)
            out["F"] = heu_vals

    class OneSolutionHeuristicTermination(Termination):

        def __init__(self, heu_vals) -> None:
            super().__init__()
            self.heu_vals = heu_vals

        def _do_continue(self, algorithm):
            F = [indiv.F for indiv in algorithm.pop]
            valid_sols = []
            i = 0
            for sol in F:
                # for each fitness result collection
                # print(f'{i} = {tuple(sol)}')
                i+=1
                sol_is_valid = True
                for heu_i in range(len(sol)):
                    heu_v = sol[heu_i]
                    heu_max = self.heu_vals[heu_i]
                    if heu_v > heu_max:
                        sol_is_valid = False
                        break
                if sol_is_valid:
                    valid_sols.append(sol)

            return len(valid_sols) == 0

    problem = MyProblem()

    # ALGORITHM
    # algorithm = GA(pop_size=20, n_offsprings=10, eliminate_duplicates=True)
    # algorithm = NSGA3(ref_dirs=X, pop_size=20, n_offsprings=10)    
    algorithm = nsga2(pop_size=20, n_offsprings=10, eliminate_duplicates=True)

    # TERMINATION
    # t2 = TimeBasedTermination(max_time=60) # TODO change the timeout to a cmd-line param
    # termination = TerminationCollection(t1, t2)
    # TODO change termination to first optimal solution + timeout
    # termination = ('n_gen', 200)
    termination = OneSolutionHeuristicTermination(heu_vals=[0, 0])

    s = random.randint(1, 10000)
    print('<    BEGIN nsga. ', end='')
    nsga_res = minimize(problem, algorithm, termination, save_history=False, verbose=True, seed=s)

    # GET OPTIMAL SOLUTION
    optimal_sols = []
    for i in range(len(nsga_res.F)):
        f = nsga_res.F[i]
        if sum(f) == 0:
            optimal_sols.append(i)

    # No solutions found for the given termination
    if optimal_sols == []:
        print('No solution found, trying again>')
        return None

    # TODO for now, we only get the first optimal solution (only one per run)
    nsga_optimal_sol = nsga_res.X[optimal_sols[0]]

    print('Solution found>')
    print('<    END nsga>')
    # STEP 3 select color-shape combos

    if args.distinct_objects:
        # TODO should be size-color combo, and not shape-color
        # the difficulty is that size affects NSGA.
        # So we need to make this "distinct" selection before NSGA
        all_shp_col_combos = []
        for shape_name in mappings['object']:
            for color_name in mappings['color']:
                all_shp_col_combos.append((shape_name, color_name))

        print(len(all_shp_col_combos))

    for i in range(num_objects):
        if args.distinct_objects:
            shape_name, color_name = random.choice(all_shp_col_combos)
        else:
            shape_name = random.choice(mappings['object'])
            color_name = random.choice(mappings['color'])
        all_shp_col_combos.remove((shape_name, color_name))
        objects_all_info[i].update({
            'name': mappings['object'][shape_name],
            'name_out': shape_name,
            'color': mappings['color'][color_name],
            'color_name': color_name
        })

        # STEP 4 also add the position and radius info
        r = radiuses[i]
        if shape_name == 'Cube':
            r /= math.sqrt(2)
        
        # get x and y from the NSGA stuff
        objects_all_info[i]['position'] = (nsga_optimal_sol[2*i], nsga_optimal_sol[2*i+1], r)

    return  objects_all_info

