from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from ..common.paths import Path

def draw_base_grid(ax: plt.Axes, game: Any):
    for i in range(0, game.grid_size + 1, game.subregion_size):
        ax.axhline(y=i, color='gray', linestyle='-', alpha=0.3)
        ax.axvline(x=i, color='gray', linestyle='-', alpha=0.3)

def draw_graph_structure(ax: plt.Axes, game: Any):
    for edge in game.graph.edges():
        ax.plot([edge[0][1], edge[1][1]], [edge[0][0], edge[1][0]], 
                color='lightgray', alpha=0.3, linewidth=1)

def draw_flows(ax: plt.Axes, flows: Dict, game: Any):
    max_flow = max(flows.values())
    for (start, end), flow in flows.items():
        if flow > 0.01:
            normalized_flow = flow / max_flow if max_flow > 0 else 0
            ax.plot([start[1], end[1]], [start[0], end[0]], 
                    color='blue', alpha=0.5,
                    linewidth=1 + 4 * normalized_flow)

def draw_od_pairs(ax: plt.Axes, game: Any):
    for (origin, dest) in game.od_pairs:
        ax.plot(origin[1], origin[0], 'go', markersize=10)
        ax.plot(dest[1], dest[0], 'ro', markersize=10)

def draw_deterministic_paths(ax: plt.Axes, paths: Dict):
    for path in paths.values():
        for i in range(len(path) - 1):
            start = path[i]
            end = path[i + 1]
            ax.plot([start[1], end[1]], [start[0], end[0]], 
                    color='red', linestyle='-',
                    linewidth=2.5, alpha=0.8)


def visualize_flows(game: Any, flows: Dict, output_path: Path) -> None:
    """
    Visualize flow distribution.
    
    Args:
        game: Routing game instance
        flows: Dictionary of edge flows
        output_path: Path object pointing to where to save the visualization
    """
    plt.figure(figsize=(12, 12))
    
    draw_base_grid(plt.gca(), game)
    draw_graph_structure(plt.gca(), game)
    draw_flows(plt.gca(), flows, game)
    draw_od_pairs(plt.gca(), game)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    plt.grid(True, which='major', color='gray', linestyle='-', alpha=0.3)
    plt.xlim(-1, game.grid_size)
    plt.ylim(game.grid_size, -1)
    
    plt.title("Abstracted Flow Distribution")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_cost_evolution(costs: List[float], output_path: Path) -> None:
    """
    Plot evolution of system cost.
    
    Args:
        costs: List of costs over iterations
        output_path: Path object pointing to where to save the plot
    """
    plt.figure(figsize=(10, 6))
    iterations = range(1, len(costs) + 1)
    
    plt.plot(iterations, costs, 'b-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Total System Cost')
    plt.title('Evolution of Total System Cost')
    plt.grid(True, alpha=0.3)
    
    if max(costs) / min(costs) > 10:
        plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_combined_animation(game: Any, all_flows: List, paths: Dict, output_path: Path) -> None:
    """
    Create animation showing flow evolution and final paths.
    
    Args:
        game: Routing game instance
        all_flows: List of flow states over iterations
        paths: Dictionary of final paths
        output_path: Path object pointing to where to save the animation
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    
    def update(frame):
        ax.clear()
        
        if frame < len(all_flows):
            iteration, flows = all_flows[frame]
        else:
            iteration, flows = all_flows[-1]
        
        draw_base_grid(ax, game)
        draw_graph_structure(ax, game)
        draw_flows(ax, flows, game)
        draw_od_pairs(ax, game)
        
        if frame == len(all_flows):
            draw_deterministic_paths(ax, paths)
            ax.plot([], [], 'go', markersize=10, label='Origins')
            ax.plot([], [], 'ro', markersize=10, label='Destinations')
            ax.plot([], [], 'r-', linewidth=2.5, label='Deterministic Paths')
            ax.legend(loc='upper right')
        
        ax.grid(True, which='major', color='gray', linestyle='-', alpha=0.3)
        ax.set_xlim(-1, game.grid_size)
        ax.set_ylim(game.grid_size, -1)
        
        title = "Abstracted Flow Distribution"
        if frame < len(all_flows):
            title += f" - Iteration {iteration}"
        else:
            title += " with Deterministic Paths"
        ax.set_title(title)
    
    anim = FuncAnimation(
        fig, update,
        frames=len(all_flows) + 1,
        interval=1000,
        blit=False
    )
    
    anim.save(output_path, writer='pillow', fps=1)
    plt.close(fig)