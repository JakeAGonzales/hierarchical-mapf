import os
import time
import numpy as np
from pathlib import Path
from src.routing_game import (
    AbstractedRoutingGame,
    RoutingGameConfig,
    extract_deterministic_paths,
    analyze_paths,
    visualize_flows,
    create_combined_animation,
    plot_cost_evolution
)

def setup_output_dirs() -> Path:
    """Create output directories if they don't exist"""
    gif_dir = Path("gifs/routing")
    gif_dir.mkdir(parents=True, exist_ok=True)
    return gif_dir

def run_routing_example():
    results = np.load("results_reg.npz")
    A_matrix = results['A']
    
    output_dir = setup_output_dirs()
    print("\n=== Running Abstracted Routing Example ===")
    
    config = RoutingGameConfig(
        grid_size=32,
        subregion_size=16,
        num_od_pairs=5,
        total_flow=1.0,
        boundary_type="full_grid"
    )
    
    print("\nInitializing routing game...")
    game = AbstractedRoutingGame(config=config, A_matrix=A_matrix)
    
    print("\nRunning Frank-Wolfe algorithm...")
    normalized_flows, costs, computation_time, all_flows = game.run_frank_wolfe(
        max_iterations=50,
        convergence_threshold=1e-1,
        sample_rate=5
    )
    
    print(f"\nOptimization completed in {computation_time:.2f} seconds")
    print(f"Final system cost: {costs[-1]:.2f}")
    
    print("\nExtracting deterministic paths...")
    paths, travel_times = extract_deterministic_paths(game, normalized_flows)
    
    print("\nAnalyzing paths...")
    path_analysis = analyze_paths(game, paths)
    
    # Print analysis results
    print("\nPath Analysis Results:")
    for od_pair, analysis in path_analysis.items():
        origin, dest = od_pair
        print(f"\nPath from {origin} to {dest}:")
        print(f" Path length: {analysis['path_length']}")
        print(f" Number of subregions: {analysis['num_subregions']}")
        print(f" Travel time: {travel_times[od_pair]:.2f}")
        print(f" Subregion sequence: {analysis['subregion_sequence']}")
    
    print("\nGenerating visualizations...")
    plot_cost_evolution(
        costs,
        output_dir / "cost_evolution.png"
    )
    
    print("\nCreating flow animation...")
    create_combined_animation(
        game,
        all_flows,
        paths,
        output_dir / "flow_evolution.gif"
    )
    
    print(f"\nVisualizations saved to {output_dir}")
    print("\nExample completed successfully!")

if __name__ == "__main__":
    run_routing_example()