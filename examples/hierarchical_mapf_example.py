import os
import time
import networkx as nx
from pathlib import Path

from src.routing_game import (
    AbstractedRoutingGame,
    RoutingGameConfig,
    extract_deterministic_paths,
    analyze_paths,
    create_combined_animation
)

from src.hierarchical_mapf import (
    GridRegion,
    HierarchicalEnvironment,
    BoundaryGoal,
    hierarchical_cbs,
    init_hcbs,
    PathVertex,
    get_solution_metrics,
    verify_solution,
    GridWorld,
    column_lattice_obstacles,
    create_solution_animation
)


def setup_output_dirs() -> Path:
    """Create output directories if they don't exist"""
    output_dir = Path("output/")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def get_subregion_coords(point: tuple, subregion_size: int) -> tuple:
    """Convert a point to its subregion coordinates"""
    return (point[0] // subregion_size, point[1] // subregion_size)

def get_region_sequence(path: list, subregion_size: int) -> list:
    """Get sequence of regions a path passes through"""
    if not path:
        return []
    
    # Get sequence of subregions, including duplicates
    sequence = [get_subregion_coords(point, subregion_size) for point in path]
    
    # Remove consecutive duplicates while preserving order
    unique_sequence = []
    for region in sequence:
        if not unique_sequence or region != unique_sequence[-1]:
            unique_sequence.append(region)
            
    return unique_sequence

def create_hierarchical_environment(config: RoutingGameConfig):
    """Create a hierarchical environment with the given configuration"""
    # Calculate grid dimensions
    nrows = config.grid_size // config.subregion_size
    ncols = config.grid_size // config.subregion_size
    
    # Create obstacles using column lattice pattern
    obstacles = column_lattice_obstacles(
        h=2,  # height of obstacle columns
        w=2,  # width of obstacle columns
        dy=1, # vertical spacing
        dx=1, # horizontal spacing
        obstacle_rows=0,
        obstacle_cols=0
    )
    
    gridworld = GridWorld((config.grid_size, config.grid_size), obstacles)
    region_graph = nx.Graph()
    # create regions
    for i in range(nrows):
        for j in range(ncols):
            # region boundaries
            start_row = i * config.subregion_size
            start_col = j * config.subregion_size
            
            region = GridRegion(
                gridworld,
                (start_row, start_col),
                (config.subregion_size, config.subregion_size)
            )
            
            region_graph.add_node((i,j), env=region)
            
            # Connect subregions to neighbors
            if i > 0:  
                boundary_edges = [
                    (u,v) for u in region.boundary 
                    for v in region_graph.nodes[(i-1,j)]['env'].boundary
                    if gridworld.contains_edge(u,v)
                ]
                region_graph.add_edge((i,j), (i-1,j), boundary=boundary_edges)
                
            if j > 0:  
                boundary_edges = [
                    (u,v) for u in region.boundary 
                    for v in region_graph.nodes[(i,j-1)]['env'].boundary
                    if gridworld.contains_edge(u,v)
                ]
                region_graph.add_edge((i,j), (i,j-1), boundary=boundary_edges)
    
    return HierarchicalEnvironment(gridworld, region_graph)

def generate_flow_based_problem(env: HierarchicalEnvironment, config: RoutingGameConfig):
    game = AbstractedRoutingGame(config)
    
    print("\nRunning Frank-Wolfe optimization...")
    normalized_flows, costs, comp_time, all_flows = game.run_frank_wolfe(
        max_iterations=50,
        convergence_threshold=1e-1,
        sample_rate=5
    )
    print(f"Optimization completed in {comp_time:.2f} seconds")

    print("\nExtracting deterministic paths...")
    flow_paths, _ = extract_deterministic_paths(game, normalized_flows)
    
    print("\nExtracting boundary points from flow paths...")
    boundary_assignments = {}
    
    def get_subregion(point):
        return (point[0] // config.subregion_size, point[1] // config.subregion_size)
    
    for idx, (origin, dest) in enumerate(game.od_pairs):
        if (origin, dest) not in flow_paths:
            continue
            
        path = flow_paths[(origin, dest)]
        agent_boundary_pairs = []
        
        for i in range(len(path)-1):
            curr_point = path[i]
            next_point = path[i+1]
            
            curr_region = get_subregion(curr_point)
            next_region = get_subregion(next_point)
            
            if curr_region != next_region:
                agent_boundary_pairs.append((curr_point, next_point))
        
        boundary_assignments[idx] = agent_boundary_pairs

    start_pos = {}
    goals = {}
    region_paths = {}
    region_boundary_points = {}
    
    for idx, (origin, dest) in enumerate(game.od_pairs):
        start_pos[idx] = origin
        goals[idx] = dest
        
        if (origin, dest) in flow_paths:
            path = flow_paths[(origin, dest)]
            sequence = get_region_sequence(path, config.subregion_size)
            region_paths[idx] = sequence
            
            if idx in boundary_assignments:
                region_boundary_points[idx] = {}
                boundary_pairs = boundary_assignments[idx]
                for i, region in enumerate(sequence[:-1]):
                    region_boundary_points[idx][region] = {
                        'entry': boundary_pairs[i][0] if i == 0 else boundary_pairs[i-1][1],
                        'exit': boundary_pairs[i][0]
                    }
                if sequence:
                    region_boundary_points[idx][sequence[-1]] = {
                        'entry': boundary_pairs[-1][1] if boundary_pairs else None,
                        'exit': None
                    }
        else:
            start_region = get_subregion(origin)
            goal_region = get_subregion(dest)
            region_paths[idx] = nx.shortest_path(env.region_graph, start_region, goal_region)
    
    x = {i: PathVertex(start_pos[i], 0) for i in range(len(game.od_pairs))}
    
    return x, goals, region_paths, game, normalized_flows, all_flows, flow_paths, region_boundary_points

def run_hierarchical_example(search_type='focal'):  # or 'astar'
    output_dir = setup_output_dirs()
    print("\n=== Running Hierarchical MAPF Example ===")
    
    config = RoutingGameConfig(
        grid_size=32,
        subregion_size=8,
        num_od_pairs=60,
        total_flow=1.0,
        boundary_type="full_grid"
    )
    
    print("\nInitializing hierarchical environment...")
    env = create_hierarchical_environment(config)
    
    print("\nGenerating flow-based problem...")
    x, goals, region_paths, game, flows, all_flows, flow_paths, region_boundary_points = generate_flow_based_problem(
        env, config
    )

    print("\nInitializing HCBS...")
    root = init_hcbs(env, x, goals, region_paths, region_boundary_points)
    
    print("\nRunning Hierarchical CBS...")
    start_time = time.time()
    solution = hierarchical_cbs(
        root,
        env,
        region_boundary_points,  
        search_type='astar',
        omega=1.0,
        maxtime=60,
        cbs_maxtime=60,
        verbose=True
    )
    comp_time = time.time() - start_time
    print(f"HCBS completed in {comp_time:.2f} seconds")
    
    if solution:
        print("\nSolution found!")
        mapf_solution = solution.make_solution()

        print("\n")
        print("*" * 50)
        print(f"SUCCESSFULLY SOLVED FOR {config.num_od_pairs} AGENTS")
        print("*" * 50)
        print("\n")
        
        # Verify solution
        print("\nVerifying solution...")
        is_valid = verify_solution(mapf_solution.paths, env)
        
        if is_valid:
            print("✓ Solution verified successfully")
            metrics = get_solution_metrics(mapf_solution.paths) 
            print("\nSolution Metrics:")
            print(f"Makespan: {metrics['makespan']}")
            print(f"Total Path Length: {metrics['total_path_length']}")
            print(f"Average Path Length: {metrics['avg_path_length']:.2f}")
        else:
            print("✗ Solution verification failed")
            return
        
        print("\nGenerating visualizations...")
        
        print("Creating MAPF animation...")
        create_solution_animation(
            env,
            mapf_solution,
            config,
            output_dir / "mapf_solution.gif"
        )
        
        print("Creating flow animation...")
        create_combined_animation(
            game,
            all_flows,
            flow_paths,
            output_dir / "flow_evolution.gif"
        )
        
        print(f"\nVisualizations saved to {output_dir}")
        
    else:
        print("\n✗ No solution found within time limit")
    
    print("\nExample completed!")

if __name__ == "__main__":
    run_hierarchical_example()