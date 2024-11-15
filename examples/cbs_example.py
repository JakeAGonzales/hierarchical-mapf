from typing import Tuple, Set, List
import random
import time
from pathlib import Path

from src.cbs import (
    conflict_based_search,
    Environment,
    GridWorld,
    MAPFProblem,
    visualize_cbs,
    calculate_solution_cost
)
from examples.utils import read_map_file, get_boundary_positions, ensure_dir  

def create_random_problem(height: int, width: int, 
                         obstacles: Set[Tuple[int, int]], 
                         num_agents: int) -> MAPFProblem:
    """Create a random MAPF problem with agents on the boundary."""
    boundary = get_boundary_positions(height, width, obstacles)
    
    if len(boundary) < 2 * num_agents:
        raise ValueError("Not enough boundary positions for the given number of agents")
    
    positions = random.sample(boundary, 2 * num_agents)
    agent_pos = positions[:num_agents]
    goals = positions[num_agents:]
 
    env = Environment((height, width), list(obstacles), agent_pos)
    return MAPFProblem(env, goals)

def run_cbs_example(map_file: str, num_agents: int, create_vis: bool, time_limit: float = 60.0):
    gif_dir = Path(__file__).parent / "gifs"
    ensure_dir(gif_dir)
    
    print(f"\nReading map file: {map_file}")
    height, width, obstacles = read_map_file(map_file)
    #height, width = 8,8
    problem = create_random_problem(height, width, obstacles, num_agents)
    
    print(f"\nRunning CBS with {num_agents} agents...")
    start_time = time.time()
    solution = conflict_based_search(problem, time_limit=time_limit)
    runtime = time.time() - start_time
    
    if solution:
        print("\nSolution found!")
        print(f"Runtime: {runtime:.2f} seconds")
        print(f"Makespan: {solution.makespan}")
        print(f"Total solution cost: {calculate_solution_cost(solution)}")
        
        if create_vis: 
            gif_path = gif_dir / f"cbs_{num_agents}_agents.gif"
            print(f"\nCreating visualization: {gif_path}")
            visualize_cbs(problem, solution, str(gif_path))
    else:
        print("\nNo solution found.")
        print(f"Runtime: {runtime:.2f} seconds")

def main():
    map_file = "maps/empty-32-32.map"  
    run_cbs_example(map_file, num_agents=32, create_vis=True)

if __name__ == "__main__":
    main()
