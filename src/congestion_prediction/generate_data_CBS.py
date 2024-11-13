import numpy as np
import pickle
import os
import multiprocessing
from typing import List, Tuple
import cbs  

def load_map_file(map_file_path: str) -> List[List[int]]:
    with open(map_file_path, 'r') as f:
        lines = f.readlines()
    
    height = int(lines[1].split()[1])
    width = int(lines[2].split()[1])
    
    obstacles = [[0 for _ in range(width)] for _ in range(height)]
    
    for i, line in enumerate(lines[4:]):
        for j, char in enumerate(line.strip()):
            if char == '@':
                obstacles[i][j] = 1
    
    return obstacles

def get_boundary_positions(size: Tuple[int, int]) -> List[Tuple[int, int]]:
    height, width = size
    boundary = []
    for i in range(height):
        boundary.append((i, 0))
        boundary.append((i, width - 1))
    for j in range(1, width - 1):
        boundary.append((0, j))
        boundary.append((height - 1, j))
    return boundary

def gen_problem(size: Tuple[int, int], n_agents: int, rng: np.random.Generator) -> cbs.MAPFProblem:
    boundary_positions = get_boundary_positions(size)
    rng.shuffle(boundary_positions)
    
    agent_pos = boundary_positions[:n_agents]
    goal_pos = boundary_positions[n_agents:2*n_agents]
    obstacle_pos = []  # No obstacles in the 32x32 empty grid

    env = cbs.Environment(size, obstacle_pos, agent_pos)
    return cbs.MAPFProblem(env, goal_pos)

def task(args):
    seed, num_examples, map_file_path = args
    rng = np.random.default_rng(seed=seed)
    obstacles = load_map_file(map_file_path)
    size = (len(obstacles), len(obstacles[0]))
    
    for i in range(num_examples):
        n_agents = rng.integers(10, 50)  # Random number of agents between lb and ub
        prob = gen_problem(size, n_agents, rng)
        soln = cbs.conflict_based_search(prob)
        if soln is not None:
            data = {'problem': prob, 'solution': soln}
            try:
                os.makedirs('data/process{}'.format(seed), exist_ok=True)
            except:
                pass
            with open(f'data/process{seed}/data{i}.pickle', 'wb') as file:
                pickle.dump(data, file)

if __name__ == "__main__":
    n_cpus = 8
    map_file_path = "empty-32-32.map"  
    total_examples = 5000
    grid_size = 8

    with multiprocessing.Pool(n_cpus) as p:
        # Make sure to use the right set of seeds
        train_seeds = [18427, 31072, 53165, 57585, 75815, 89991, 92318, 98555] 
        test_seeds = [24601, 42069, 87654, 13579, 99999, 54321, 11111, 77777] 
        p.map(task, [(test_seeds, total_examples // n_cpus, map_file_path) for test_seeds in test_seeds])
        #p.map(task, [(train_seeds, total_examples // n_cpus, map_file_path) for train_seeds in train_seeds])