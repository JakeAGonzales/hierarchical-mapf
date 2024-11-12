#!/usr/bin/env python3

from typing import Dict, Tuple
import time
import numpy as np

from src.hierarchical_mapf import (
    HCBS,
    HCBSConfig,
    SimpleRegionalEnvironment,
    create_solution_animation,
    verify_solution
)

from src.routing_game import (
    AbstractedRoutingGame,
    RoutingGameConfig,
    extract_deterministic_paths,
    analyze_paths,
    create_combined_animation
)

def run_flow_guided_rcbs(
    num_agents: int = 5,
    world_size: Tuple[int, int] = (32, 32),
    region_size: Tuple[int, int] = (8, 8)
) -> Dict:
    """
    Run complete flow-guided RCBS example
    
    Args:
        num_agents: Number of agents to plan for
        world_size: Size of the world grid (height, width)
        region_size: Size of each region (height, width)
    """
    print("\n=== Running Flow-Guided RCBS Example ===")
    
    # 1. Set up the routing game
    print("\nInitializing routing game...")
    game_config = RoutingGameConfig(
        grid_size=world_size[0],
        subregion_size=region_size[0],
        num_od_pairs=num_agents,
        boundary_type='full_grid'
    )
    game = AbstractedRoutingGame(game_config)
    
    # 2. Run Frank-Wolfe to get flows
    print("\nComputing flow patterns...")
    normalized_flows, costs, runtime, all_flows = game.run_frank_wolfe(
        max_iterations=50,
        convergence_threshold=1e-1,
        sample_rate=5
    )
    
    # 3. Extract deterministic paths and analyze
    flow_paths, _ = extract_deterministic_paths(game, normalized_flows)
    path_analysis = analyze_paths(game, flow_paths)
    
    # 4. Create environment and initial state
    print("\nSetting up MAPF environment...")
    env = SimpleRegionalEnvironment(world_size, region_size)
    
    # Convert routing game paths to RCBS format
    mapf_inputs = convert_flow_paths_to_mapf(game, flow_paths, path_analysis)
    start_positions, goals, region_paths = mapf_inputs
    
    # 5. Initialize and run HCBS
    print("\nRunning HCBS...")
    start_time = time.time()
    
    # Create config and solver
    config = HCBSConfig(
        max_time=60.0,
        cbs_time=30.0,
        omega=1.0,
        verbose=True
    )
    solver = HCBS(config)
    
    # Run HCBS
    solution = solver.solve(
        env=env,
        x=start_positions,
        final_goals=goals,
        region_paths=region_paths
    )
    
    runtime = time.time() - start_time
    print(f"\nTotal HCBS Runtime: {runtime:.2f} seconds")
    
    # 6. Create visualizations
    if solution:
        print("\nCreating visualizations...")
        # Convert solution to proper format if needed
        mapf_solution = solution.make_solution()
        create_solution_animation(env.gridworld, mapf_solution, region_size)
        create_combined_animation(game, all_flows, flow_paths, 'flows_and_paths.gif')
        
        # Verify solution
        print("\nVerifying solution...")
        is_valid = verify_solution(mapf_solution, env.gridworld)
        
        print("\n=== Final Validation Results ===")
        if is_valid:
            print("✓ Solution is valid!")
            print(f"  - Total cost: {solution.cost}")
        else:
            print("✗ Solution has validation issues")
    else:
        print("\n✗ No solution found within time limit")
    
    return {
        'solution': solution,
        'environment': env,
        'game': game,
        'flows': normalized_flows,
        'runtime': runtime
    }

def convert_flow_paths_to_mapf(
    game: AbstractedRoutingGame,
    flow_paths: Dict,
    path_analysis: Dict
) -> Tuple[Dict, Dict, Dict]:
    """
    Convert routing game paths to MAPF format
    
    Args:
        game: Routing game instance
        flow_paths: Dictionary of paths from routing game
        path_analysis: Analysis of paths including subregion sequences
        
    Returns:
        Tuple of (start_positions, goals, region_paths)
    """
    start_positions = {}
    goals = {}
    region_paths = {}
    
    for agent_id, ((origin, dest), _) in enumerate(game.od_pairs.items()):
        # Set start and goal
        start_positions[agent_id] = origin
        goals[agent_id] = dest
        
        # Get region path
        if (origin, dest) in path_analysis:
            region_paths[agent_id] = path_analysis[(origin, dest)]['subregion_sequence']
        else:
            # Fallback to direct path if flow path not found
            start_region = (origin[0] // game.subregion_size, 
                          origin[1] // game.subregion_size)
            goal_region = (dest[0] // game.subregion_size,
                          dest[1] // game.subregion_size)
            region_paths[agent_id] = [start_region, goal_region]
    
    return start_positions, goals, region_paths

if __name__ == "__main__":
    # Run with different numbers of agents
    agent_counts = [5, 10]
    
    for num_agents in agent_counts:
        print(f"\n{'='*20} Testing {num_agents} Agents {'='*20}")
        results = run_flow_guided_rcbs(num_agents=num_agents)
        
        if results['solution']:
            print(f"Success! Solution cost: {results['solution'].cost}")
        else:
            print("Failed to find solution")
        print(f"Runtime: {results['runtime']:.2f} seconds")