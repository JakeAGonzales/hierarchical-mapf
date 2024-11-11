from typing import Dict, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from ..common import GridWorld
from .core import HCBSNode

def create_solution_animation(
    env: GridWorld, 
    solution: Dict,
    region_size: Tuple[int, int],
    filename: str = "hierarchical_solution.gif"
) -> None:
    fig, ax = plt.subplots(figsize=(10, 10))
    
    def setup_grid():
        # Draw region boundaries
        for i in range(0, env.size[0] + 1, region_size[0]):
            ax.axhline(y=i-0.5, color='gray', linestyle='-', alpha=0.5)
            ax.axvline(x=i-0.5, color='gray', linestyle='-', alpha=0.5)
        
        # Setup coordinate grid
        ax.grid(True, color='lightgray', linewidth=0.5, alpha=0.3)
        ax.set_xticks(np.arange(-0.5, env.size[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, env.size[0], 1), minor=True)
        ax.set_xlim(-0.5, env.size[1] - 0.5)
        ax.set_ylim(env.size[0] - 0.5, -0.5)

    def init_markers():
        markers = {}
        paths = {}
        histories = {}
        
        for agent_id in solution:
            # Start and goal positions
            start_pos = solution[agent_id][0].pos
            goal_pos = solution[agent_id][-1].pos
            
            ax.plot(start_pos[1], start_pos[0], 'o', color='green', 
                   markersize=12, alpha=0.7)
            ax.plot(goal_pos[1], goal_pos[0], 's', color='red', 
                   markersize=12, alpha=0.7)
            
            # Agent marker and path
            markers[agent_id], = ax.plot([], [], 'o', color='navy', 
                                       markersize=8, alpha=0.8)
            paths[agent_id], = ax.plot([], [], '-', color='navy', 
                                     linewidth=1.5, alpha=0.3)
            histories[agent_id] = {'x': [], 'y': []}
        
        return markers, paths, histories

    def update(frame):
        ax.set_title(f'Time Step: {frame}')
        
        for agent_id, path in solution.items():
            if frame < len(path):
                pos = path[frame].pos
                markers[agent_id].set_data([pos[1]], [pos[0]])
                histories[agent_id]['x'].append(pos[1])
                histories[agent_id]['y'].append(pos[0])
            else:
                pos = path[-1].pos
                markers[agent_id].set_data([pos[1]], [pos[0]])
            
            paths[agent_id].set_data(
                histories[agent_id]['x'], 
                histories[agent_id]['y']
            )
        
        return list(markers.values()) + list(paths.values())

    setup_grid()
    markers, paths, histories = init_markers()
    max_timesteps = max(len(path) for path in solution.values())
    
    anim = FuncAnimation(
        fig, update,
        frames=max_timesteps,
        interval=200,
        blit=True
    )
    
    anim.save(filename, writer='pillow', fps=5)
    plt.close()

def verify_solution(solution: Dict, env: GridWorld) -> bool:
    collisions = _check_collisions(solution)
    path_validity = _check_path_validity(solution, env)
    
    if collisions or not path_validity:
        print("\nSolution Verification Failed:")
        if collisions:
            print("- Found agent collisions")
        if not path_validity:
            print("- Found invalid or discontinuous paths")
        return False
        
    return True

def _check_collisions(solution: Dict) -> bool:
    occupied = {}
    
    for agent_id, path in solution.items():
        for vertex in path:
            t = vertex.t
            pos = vertex.pos
            
            if t not in occupied:
                occupied[t] = {pos: agent_id}
            elif pos in occupied[t]:
                other_agent = occupied[t][pos]
                print(f"\nCollision at t={t} between agents {agent_id} and {other_agent}")
                return True
            else:
                occupied[t][pos] = agent_id
                
            # Check edge collisions
            if t > 0:
                prev_pos = path[t-1].pos
                for other_id, other_path in solution.items():
                    if other_id != agent_id and t < len(other_path):
                        if (other_path[t].pos == prev_pos and 
                            other_path[t-1].pos == pos):
                            print(f"\nEdge collision at t={t} between agents {agent_id} and {other_id}")
                            return True
    
    return False

def _check_path_validity(solution: Dict, env: GridWorld) -> bool:
    for agent_id, path in solution.items():
        # Check node validity
        for vertex in path:
            if not env.contains_node(vertex.pos):
                print(f"\nInvalid position for agent {agent_id} at t={vertex.t}: {vertex.pos}")
                return False
        
        # Check path continuity
        for i in range(len(path) - 1):
            current = path[i]
            next_vertex = path[i + 1]
            
            if next_vertex.t != current.t + 1:
                print(f"\nTime discontinuity for agent {agent_id} at t={current.t}")
                return False
                
            if not env.contains_edge(current.pos, next_vertex.pos):
                print(f"\nInvalid move for agent {agent_id} at t={current.t}")
                return False
    
    return True