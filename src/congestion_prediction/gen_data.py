import numpy as np
import pickle
import os
import multiprocessing
from typing import List, Tuple
import cbs

# gen data script with empty small subregions like 8x8 or 16x16

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
    obstacle_pos = []  # No obstacles
    env = cbs.Environment(size, obstacle_pos, agent_pos)
    return cbs.MAPFProblem(env, goal_pos)

def task(args):
    seed, num_examples, grid_size, agent_range = args
    rng = np.random.default_rng(seed=seed)
    size = (grid_size, grid_size)
    
    for i in range(num_examples):
        n_agents = rng.integers(agent_range[0], agent_range[1])
        prob = gen_problem(size, n_agents, rng)
        soln = cbs.conflict_based_search(prob)
        
        if soln is not None:
            data = {'problem': prob, 'solution': soln}
            try:
                os.makedirs(f'data/process{seed}_{grid_size}x{grid_size}', exist_ok=True)
            except:
                pass
            with open(f'data/process{seed}_{grid_size}x{grid_size}/data{i}.pickle', 'wb') as file:
                pickle.dump(data, file)

if __name__ == "__main__":
    # Configuration
    n_cpus = 8
    total_examples = 2000
    grid_size = 8  # w x h
    
    agent_range = (2, 10) 

    test_seeds = [24601, 42069, 87654, 13579, 99999, 54321, 11111, 77777]
    train_seeds = [18427, 31072, 53165, 57585, 75815, 89991, 92318, 98555]
    
    print(f"Generating data for {grid_size}x{grid_size} grid...")
    with multiprocessing.Pool(n_cpus) as p:
        # Generate test data
        p.map(task, [(seed, total_examples // n_cpus, grid_size, agent_range) 
                   for seed in test_seeds])
        
        # Generate train data (uncomment to use)
        #p.map(task, [(seed, total_examples // n_cpus, grid_size, agent_range) 
         #            for seed in train_seeds])
    
    print(f"Completed {grid_size}x{grid_size} grid")


    # Notes: for 8x8 we used 5000 train examples, 1000 test examples, and agents from range(2,12)
    # for 16x16 we used 5000/1000 train/test and agnets from range(4,16) 