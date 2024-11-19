import os
import time
import numpy as np
import networkx as nx
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from collections import defaultdict

from src.routing_game import (
    AbstractedRoutingGame,
    RoutingGameConfig,
    extract_deterministic_paths,
)

from src.hierarchical_mapf import (
    GridRegion,
    HierarchicalEnvironment,
    hierarchical_cbs,
    init_hcbs,
    PathVertex,
    get_solution_metrics,
    verify_solution,
    GridWorld,
    column_lattice_obstacles,
)

from cbs import (
    conflict_based_search,
    MAPFProblem,
    Environment,
    MAPFSolution
)

# runs a comparison on CBS v. HCBS (only considers successful runs or runs that timeout in T = 60s)

class ComparisonResults:
    def __init__(self):
        self.metrics = defaultdict(lambda: {
            'hcbs_success': [],
            'cbs_success': [],
            'hcbs_times': [],
            'cbs_times': [],
            'routing_times': [],
            'hcbs_path_costs': [],
            'cbs_path_costs': [],
            'hcbs_makespans': [],
            'cbs_makespans': []
        })

    def add_trial(self, num_agents: int, hcbs_success: bool, cbs_success: bool,
                 hcbs_time: float, cbs_time: float, routing_time: float,
                 hcbs_path_cost: float, cbs_path_cost: float,
                 hcbs_makespan: int, cbs_makespan: int):
        metrics = self.metrics[num_agents]
        metrics['hcbs_success'].append(hcbs_success)
        metrics['cbs_success'].append(cbs_success)
        metrics['hcbs_times'].append(hcbs_time)
        metrics['cbs_times'].append(cbs_time)
        metrics['routing_times'].append(routing_time)
        metrics['hcbs_path_costs'].append(hcbs_path_cost)
        metrics['cbs_path_costs'].append(cbs_path_cost)
        metrics['hcbs_makespans'].append(hcbs_makespan)
        metrics['cbs_makespans'].append(cbs_makespan)

def calculate_solution_cost(solution: MAPFSolution) -> int:
    if not solution or not solution.paths:
        return 0
    return sum(len(path) - 1 for path in solution.paths)

def create_hierarchical_environment(config: RoutingGameConfig) -> HierarchicalEnvironment:
    """Create hierarchical environment with no obstacles"""
    nrows = config.grid_size // config.subregion_size
    ncols = config.grid_size // config.subregion_size
    
    # Create empty gridworld with no obstacles
    obstacles = []  # Empty list for no obstacles
    gridworld = GridWorld((config.grid_size, config.grid_size), obstacles)
    region_graph = nx.Graph()
    
    for i in range(nrows):
        for j in range(ncols):
            start_row = i * config.subregion_size
            start_col = j * config.subregion_size
            
            region = GridRegion(gridworld, (start_row, start_col),
                              (config.subregion_size, config.subregion_size))
            region_graph.add_node((i,j), env=region)
            
            if i > 0:
                boundary_edges = [(u,v) for u in region.boundary 
                                for v in region_graph.nodes[(i-1,j)]['env'].boundary
                                if gridworld.contains_edge(u,v)]
                region_graph.add_edge((i,j), (i-1,j), boundary=boundary_edges)
                
            if j > 0:
                boundary_edges = [(u,v) for u in region.boundary 
                                for v in region_graph.nodes[(i,j-1)]['env'].boundary
                                if gridworld.contains_edge(u,v)]
                region_graph.add_edge((i,j), (i,j-1), boundary=boundary_edges)
    
    return HierarchicalEnvironment(gridworld, region_graph)

def run_comparison(num_agents: int, num_trials: int = 10, time_limit: float = 60.0) -> ComparisonResults:
    results = ComparisonResults()
    valid_trials = 0
    attempts = 0
    max_attempts = 50  
    
    config = RoutingGameConfig(
        grid_size=32,
        subregion_size=8,
        num_od_pairs=num_agents,
        total_flow=1.0,
        boundary_type="all_boundaries"
    )
    
    while valid_trials < num_trials and attempts < max_attempts:
        attempts += 1
        print(f"\nTrial attempt {attempts} for {num_agents} agents (Valid trials: {valid_trials})")
        
        # Create environment
        env = create_hierarchical_environment(config)
        
        # Run routing game and time it
        routing_start = time.time()
        game = AbstractedRoutingGame(config)
        normalized_flows, _, routing_time, _ = game.run_frank_wolfe(
            max_iterations=50,
            convergence_threshold=1e-1,
            sample_rate=5
        )
        flow_paths, _ = extract_deterministic_paths(game, normalized_flows)
        routing_time = time.time() - routing_start
        
        # Extract OD pairs and goals
        od_pairs = game.od_pairs
        start_pos = {i: od_pairs[i][0] for i in range(num_agents)}
        goals = {i: od_pairs[i][1] for i in range(num_agents)}
        
        # Run HCBS
        x = {i: PathVertex(start_pos[i], 0) for i in range(num_agents)}
        region_paths = {}
        region_boundary_points = {}
        
        # Initialize region paths (simplified for comparison)
        for i in range(num_agents):
            start = start_pos[i]
            goal = goals[i]
            start_region = (start[0] // config.subregion_size, start[1] // config.subregion_size)
            goal_region = (goal[0] // config.subregion_size, goal[1] // config.subregion_size)
            region_paths[i] = nx.shortest_path(env.region_graph, start_region, goal_region)
        
        root = init_hcbs(env, x, goals, region_paths, region_boundary_points)
        
        # Time HCBS
        hcbs_start = time.time()
        hcbs_solution = hierarchical_cbs(
            root,
            env,
            region_boundary_points,
            search_type='astar',
            omega=1.0,
            maxtime=time_limit,
            cbs_maxtime=time_limit,
            verbose=False
        )
        hcbs_time = time.time() - hcbs_start
        
        # Check HCBS solution validity
        hcbs_success = False
        hcbs_path_cost = float('inf')
        hcbs_makespan = float('inf')
        
        print(f"HCBS completed in {hcbs_time:.2f}s")
        
        if hcbs_solution:
            try:
                print("HCBS found a solution, checking validity...")
                mapf_solution = hcbs_solution.make_solution()
                if verify_solution(mapf_solution.paths, env):
                    hcbs_success = True
                    metrics = get_solution_metrics(mapf_solution.paths)
                    hcbs_path_cost = metrics['avg_path_length']
                    hcbs_makespan = metrics['makespan']
                    print("Solution verified successfully")
                else:
                    print("Solution verification failed")
                    continue
            except Exception as e:
                print(f"Path sequential test failed: {str(e)}")
                continue
        else:
            print(f"HCBS failed to find solution")
        
        # Run CBS with same OD pairs
        print("Running CBS...")
        env_cbs = Environment((config.grid_size, config.grid_size), [], list(start_pos.values()))
        problem = MAPFProblem(env_cbs, list(goals.values()))
        
        cbs_start = time.time()
        cbs_solution = conflict_based_search(problem, time_limit=time_limit)
        cbs_time = time.time() - cbs_start
        
        cbs_success = False
        cbs_path_cost = float('inf')
        cbs_makespan = float('inf')
        
        if cbs_solution:
            cbs_success = True
            cbs_path_cost = calculate_solution_cost(cbs_solution) / num_agents
            cbs_makespan = cbs_solution.makespan
            print("CBS found solution")
        else:
            print("CBS failed to find solution")
        
        # Record results
        results.add_trial(
            num_agents,
            hcbs_success,
            cbs_success,
            hcbs_time,
            cbs_time,
            routing_time,
            hcbs_path_cost,
            cbs_path_cost,
            hcbs_makespan,
            cbs_makespan
        )
        
        valid_trials += 1
        print(f"Completed valid trial {valid_trials}")
    
    if attempts >= max_attempts:
        print(f"Warning: Reached maximum attempts ({max_attempts}) for {num_agents} agents")
        
    return results

def save_results(results: ComparisonResults, output_path: str):
    with open(output_path, 'w') as f:
        f.write("MAPF Comparison Results\n")
        f.write("=====================\n\n")
        
        for num_agents in sorted(results.metrics.keys()):
            metrics = results.metrics[num_agents]
            
            f.write(f"\nResults for {num_agents} agents:\n")
            f.write("-" * 30 + "\n")
            
            # Success rates
            hcbs_success_rate = np.mean(metrics['hcbs_success']) * 100
            cbs_success_rate = np.mean(metrics['cbs_success']) * 100
            f.write(f"HCBS Success Rate: {hcbs_success_rate:.1f}%\n")
            f.write(f"CBS Success Rate: {cbs_success_rate:.1f}%\n")
            
            # Times - only consider successful runs
            hcbs_success_mask = np.array(metrics['hcbs_success'])
            cbs_success_mask = np.array(metrics['cbs_success'])
            
            hcbs_success_times = np.array(metrics['hcbs_times'])[hcbs_success_mask]
            cbs_success_times = np.array(metrics['cbs_times'])[cbs_success_mask]
            routing_times = np.array(metrics['routing_times'])
            
            # Calculate statistics only if there are successful runs
            if len(hcbs_success_times) > 0:
                hcbs_mean_time = np.mean(hcbs_success_times)
                hcbs_std_time = np.std(hcbs_success_times)
            else:
                hcbs_mean_time = float('inf')
                hcbs_std_time = float('nan')
                
            if len(cbs_success_times) > 0:
                cbs_mean_time = np.mean(cbs_success_times)
                cbs_std_time = np.std(cbs_success_times)
            else:
                cbs_mean_time = float('inf')
                cbs_std_time = float('nan')
            
            routing_mean_time = np.mean(routing_times)
            routing_std_time = np.std(routing_times)
            
            f.write(f"\nComputation Times:\n")
            f.write(f"HCBS: {hcbs_mean_time:.2f} ± {hcbs_std_time:.2f}s\n")
            f.write(f"CBS: {cbs_mean_time:.2f} ± {cbs_std_time:.2f}s\n")
            f.write(f"Routing Game: {routing_mean_time:.2f} ± {routing_std_time:.2f}s\n")
            
            # Path costs - only consider successful runs
            hcbs_success_costs = np.array(metrics['hcbs_path_costs'])[hcbs_success_mask]
            cbs_success_costs = np.array(metrics['cbs_path_costs'])[cbs_success_mask]
            
            if len(hcbs_success_costs) > 0:
                hcbs_mean_cost = np.mean(hcbs_success_costs[hcbs_success_costs != float('inf')])
                hcbs_std_cost = np.std(hcbs_success_costs[hcbs_success_costs != float('inf')])
            else:
                hcbs_mean_cost = float('inf')
                hcbs_std_cost = float('nan')
                
            if len(cbs_success_costs) > 0:
                cbs_mean_cost = np.mean(cbs_success_costs[cbs_success_costs != float('inf')])
                cbs_std_cost = np.std(cbs_success_costs[cbs_success_costs != float('inf')])
            else:
                cbs_mean_cost = float('inf')
                cbs_std_cost = float('nan')
            
            f.write(f"\nAverage Path Costs:\n")
            if hcbs_mean_cost != float('inf'):
                f.write(f"HCBS: {hcbs_mean_cost:.2f} ± {hcbs_std_cost:.2f}\n")
            else:
                f.write("HCBS: No successful solutions\n")
                
            if cbs_mean_cost != float('inf'):
                f.write(f"CBS: {cbs_mean_cost:.2f} ± {cbs_std_cost:.2f}\n")
            else:
                f.write("CBS: No successful solutions\n")
            
            f.write("\n" + "=" * 50 + "\n")

def save_results(results: ComparisonResults, output_path: str):
    with open(output_path, 'w') as f:
        f.write("MAPF Comparison Results\n")
        f.write("=====================\n\n")
        
        for num_agents in sorted(results.metrics.keys()):
            metrics = results.metrics[num_agents]
            
            f.write(f"\nResults for {num_agents} agents:\n")
            f.write("-" * 30 + "\n")
            
            # Success rates
            hcbs_success_rate = np.mean(metrics['hcbs_success']) * 100
            cbs_success_rate = np.mean(metrics['cbs_success']) * 100
            f.write(f"HCBS Success Rate: {hcbs_success_rate:.1f}%\n")
            f.write(f"CBS Success Rate: {cbs_success_rate:.1f}%\n")
            
            # Times - only consider successful runs
            hcbs_success_mask = np.array(metrics['hcbs_success'])
            cbs_success_mask = np.array(metrics['cbs_success'])
            
            hcbs_success_times = np.array(metrics['hcbs_times'])[hcbs_success_mask]
            cbs_success_times = np.array(metrics['cbs_times'])[cbs_success_mask]
            routing_times = np.array(metrics['routing_times'])
            
            # Calculate statistics only if there are successful runs
            if len(hcbs_success_times) > 0:
                hcbs_mean_time = np.mean(hcbs_success_times)
                hcbs_std_time = np.std(hcbs_success_times)
            else:
                hcbs_mean_time = float('inf')
                hcbs_std_time = float('nan')
                
            if len(cbs_success_times) > 0:
                cbs_mean_time = np.mean(cbs_success_times)
                cbs_std_time = np.std(cbs_success_times)
            else:
                cbs_mean_time = float('inf')
                cbs_std_time = float('nan')
            
            routing_mean_time = np.mean(routing_times)
            routing_std_time = np.std(routing_times)
            
            f.write(f"\nComputation Times:\n")
            f.write(f"HCBS: {hcbs_mean_time:.2f} ± {hcbs_std_time:.2f}s\n")
            f.write(f"CBS: {cbs_mean_time:.2f} ± {cbs_std_time:.2f}s\n")
            f.write(f"Routing Game: {routing_mean_time:.2f} ± {routing_std_time:.2f}s\n")
            
            # Path costs - only consider successful runs
            hcbs_success_costs = np.array(metrics['hcbs_path_costs'])[hcbs_success_mask]
            cbs_success_costs = np.array(metrics['cbs_path_costs'])[cbs_success_mask]
            
            if len(hcbs_success_costs) > 0:
                hcbs_mean_cost = np.mean(hcbs_success_costs[hcbs_success_costs != float('inf')])
                hcbs_std_cost = np.std(hcbs_success_costs[hcbs_success_costs != float('inf')])
            else:
                hcbs_mean_cost = float('inf')
                hcbs_std_cost = float('nan')
                
            if len(cbs_success_costs) > 0:
                cbs_mean_cost = np.mean(cbs_success_costs[cbs_success_costs != float('inf')])
                cbs_std_cost = np.std(cbs_success_costs[cbs_success_costs != float('inf')])
            else:
                cbs_mean_cost = float('inf')
                cbs_std_cost = float('nan')
            
            f.write(f"\nAverage Path Costs:\n")
            if hcbs_mean_cost != float('inf'):
                f.write(f"HCBS: {hcbs_mean_cost:.2f} ± {hcbs_std_cost:.2f}\n")
            else:
                f.write("HCBS: No successful solutions\n")
                
            if cbs_mean_cost != float('inf'):
                f.write(f"CBS: {cbs_mean_cost:.2f} ± {cbs_std_cost:.2f}\n")
            else:
                f.write("CBS: No successful solutions\n")
            
            f.write("\n" + "=" * 50 + "\n")

def plot_results(results: ComparisonResults, output_dir: str):
    agent_counts = sorted(results.metrics.keys())
    
    # Success Rate Plot
    plt.figure(figsize=(10, 6))
    hcbs_success = [np.mean(results.metrics[n]['hcbs_success']) * 100 for n in agent_counts]
    cbs_success = [np.mean(results.metrics[n]['cbs_success']) * 100 for n in agent_counts]
    
    plt.plot(agent_counts, hcbs_success, 'b-o', label='HCBS')
    plt.plot(agent_counts, cbs_success, 'r-o', label='CBS')
    plt.xlabel('Number of Agents')
    plt.ylabel('Success Rate (%)')
    plt.title('Success Rate vs Number of Agents')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'success_rate.png'))
    plt.close()
    
    # Runtime Plot - Use only successful trials
    plt.figure(figsize=(10, 6))
    hcbs_times = []
    hcbs_total_times = []
    cbs_times = []
    
    for n in agent_counts:
        metrics = results.metrics[n]
        hcbs_success_mask = np.array(metrics['hcbs_success'])
        cbs_success_mask = np.array(metrics['cbs_success'])
        
        # Calculate mean times for successful trials only
        if any(hcbs_success_mask):
            hcbs_mean_time = np.mean(np.array(metrics['hcbs_times'])[hcbs_success_mask])
            routing_mean_time = np.mean(np.array(metrics['routing_times'])[hcbs_success_mask])
            hcbs_times.append(hcbs_mean_time)
            hcbs_total_times.append(hcbs_mean_time + routing_mean_time)
        else:
            hcbs_times.append(np.nan)
            hcbs_total_times.append(np.nan)
            
        if any(cbs_success_mask):
            cbs_times.append(np.mean(np.array(metrics['cbs_times'])[cbs_success_mask]))
        else:
            cbs_times.append(np.nan)
    
    plt.plot(agent_counts, hcbs_total_times, 'b-o', label='HCBS + Routing')
    plt.plot(agent_counts, hcbs_times, 'g-o', label='HCBS Only')
    plt.plot(agent_counts, cbs_times, 'r-o', label='CBS')
    plt.xlabel('Number of Agents')
    plt.ylabel('Runtime (seconds)')
    plt.title('Runtime vs Number of Agents (Successful Trials Only)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'runtime.png'))
    plt.close()
    
    # Path Cost Plot - Use only successful trials
    plt.figure(figsize=(10, 6))
    hcbs_costs = []
    cbs_costs = []
    
    for n in agent_counts:
        metrics = results.metrics[n]
        hcbs_success_mask = np.array(metrics['hcbs_success'])
        cbs_success_mask = np.array(metrics['cbs_success'])
        
        # Calculate mean costs for successful trials only
        if any(hcbs_success_mask):
            hcbs_success_costs = np.array(metrics['hcbs_path_costs'])[hcbs_success_mask]
            hcbs_success_costs = hcbs_success_costs[hcbs_success_costs != float('inf')]
            if len(hcbs_success_costs) > 0:
                hcbs_costs.append(np.mean(hcbs_success_costs))
            else:
                hcbs_costs.append(np.nan)
        else:
            hcbs_costs.append(np.nan)
            
        if any(cbs_success_mask):
            cbs_success_costs = np.array(metrics['cbs_path_costs'])[cbs_success_mask]
            cbs_success_costs = cbs_success_costs[cbs_success_costs != float('inf')]
            if len(cbs_success_costs) > 0:
                cbs_costs.append(np.mean(cbs_success_costs))
            else:
                cbs_costs.append(np.nan)
        else:
            cbs_costs.append(np.nan)
    
    plt.plot(agent_counts, hcbs_costs, 'b-o', label='HCBS')
    plt.plot(agent_counts, cbs_costs, 'r-o', label='CBS')
    plt.xlabel('Number of Agents')
    plt.ylabel('Average Path Cost')
    plt.title('Average Path Cost vs Number of Agents (Successful Trials Only)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'path_cost.png'))
    plt.close()

def main():
    output_dir = Path("comparison_results")
    output_dir.mkdir(exist_ok=True)
    
    # Run comparisons for different agent counts
    all_results = ComparisonResults()
    agent_counts = [10, 20, 30, 40, 50]
    
    for num_agents in agent_counts:
        print(f"\nTesting with {num_agents} agents...")
        results = run_comparison(num_agents)
        for k, v in results.metrics.items():
            all_results.metrics[k] = v
    
    # Save and plot results
    save_results(all_results, output_dir / "comparison_results.txt")
    plot_results(all_results, output_dir)
    
    print("\nComparison completed! Results saved to comparison_results/")

if __name__ == "__main__":
    main()