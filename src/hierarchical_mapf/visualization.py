import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from .mapf_utils import MAPFSolution, Path

from src.routing_game import (
    RoutingGameConfig,
)

from src.hierarchical_mapf import (
    HierarchicalEnvironment,
)
def create_solution_animation(env: HierarchicalEnvironment, 
                            solution: MAPFSolution, 
                            config: RoutingGameConfig, 
                            output_path: Path) -> None:
    """Create and save a GIF visualization of the MAPF solution with:
    - Explicit subregions
    - Green origin markers
    - Red destination markers
    - Dark blue agents with path trajectories
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    for i in range(0, config.grid_size + 1, config.subregion_size):
        ax.axhline(y=i-0.5, color='gray', linestyle='-', alpha=0.5)
        ax.axvline(x=i-0.5, color='gray', linestyle='-', alpha=0.5)
    
    ax.grid(True, color='lightgray', linewidth=0.5, alpha=0.3)
    ax.set_xticks(np.arange(-0.5, config.grid_size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, config.grid_size, 1), minor=True)
    ax.set_xlim(-0.5, config.grid_size - 0.5)
    ax.set_ylim(config.grid_size - 0.5, -0.5)  # Invert y-axis
    
    if env.gridworld.obstacles:
        obs_y, obs_x = zip(*env.gridworld.obstacles)
        ax.scatter(obs_x, obs_y, color='black', marker='s', s=100)
    
    for agent_id in solution.paths:
        start_pos = solution.paths[agent_id][0].pos
        goal_pos = solution.paths[agent_id][-1].pos
        ax.plot(start_pos[1], start_pos[0], 'o', color='green', markersize=12, alpha=0.7)
        ax.plot(goal_pos[1], goal_pos[0], 's', color='red', markersize=12, alpha=0.7)
    
    agents = {}
    path_lines = {}
    path_histories = {agent_id: {'x': [], 'y': []} for agent_id in solution.paths}
    
    agent_color = '#000080'  # Navy blue
    
    for agent_id in solution.paths:
        agents[agent_id], = ax.plot([], [], 'o', color=agent_color, 
                                  markersize=8, alpha=0.8)
        path_lines[agent_id], = ax.plot([], [], '-', color=agent_color, 
                                      linewidth=1.5, alpha=0.3)
    
    def update(frame):
        ax.set_title(f'Time Step: {frame}')

        for agent_id, path in solution.paths.items():
            if frame < len(path):
                pos = path[frame].pos
                agents[agent_id].set_data([pos[1]], [pos[0]])
                
                path_histories[agent_id]['x'].append(pos[1])
                path_histories[agent_id]['y'].append(pos[0])
            else:
                pos = path[-1].pos
                agents[agent_id].set_data([pos[1]], [pos[0]])
            
            path_lines[agent_id].set_data(path_histories[agent_id]['x'], 
                                        path_histories[agent_id]['y'])
        
        artists = list(agents.values()) + list(path_lines.values())
        return artists
    
    anim = FuncAnimation(
        fig, 
        update,
        frames=solution.makespan,
        interval=200,
        blit=True
    )
    
    anim.save(output_path, writer='pillow', fps=5)
    plt.close()
    
    print(f"Animation saved as {output_path}")