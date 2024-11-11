import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List, Dict, Tuple, Optional

from .environment import Environment
from .problem import MAPFProblem, MAPFSolution

def calculate_solution_cost(solution: MAPFSolution) -> int:
    if not solution or not solution.paths:
        return 0
    return sum(len(path) - 1 for path in solution.paths)

def visualize_cbs(
    problem: MAPFProblem, 
    solution: MAPFSolution, 
    filename: str = 'cbs_visualization.gif'
) -> None:
    env = problem.env
    height, width = env.size
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    num_agents = len(problem.goals)
    colors = plt.cm.rainbow(np.linspace(0, 1, num_agents))
    
    def init():
        ax.clear()
        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(height - 0.5, -0.5)
        ax.set_aspect('equal')
        ax.grid(True, which='both', color='lightgray', linestyle='-', linewidth=0.5)
        for obs in env.obstacle_pos:
            ax.add_patch(plt.Rectangle((obs[1]-0.5, obs[0]-0.5), 1, 1, 
                                     facecolor='gray', edgecolor='none'))
        return []
    
    def update(frame):
        ax.clear()
        init()
        
        for i, path in enumerate(solution.paths):
            # Plot path trail
            if frame > 0:
                path_coords = [(v.pos[1], v.pos[0]) for v in path[:frame+1]]
                ax.plot(*zip(*path_coords), color=colors[i], linewidth=2, alpha=0.5)
            
            if frame < len(path):
                pos = path[frame].pos
                ax.add_patch(plt.Rectangle((pos[1]-0.4, pos[0]-0.4), 0.8, 0.8, 
                                         facecolor=colors[i], edgecolor='none'))
            
            goal = problem.goals[i]
            ax.add_patch(plt.Rectangle((goal[1]-0.4, goal[0]-0.4), 0.8, 0.8, 
                                     fill=False, edgecolor=colors[i], linewidth=2))
        
        total_cost = calculate_solution_cost(solution)
        ax.set_title(f'Step {frame} | Total Solution Cost: {total_cost}')
        return []
    
    if not solution.paths or all(len(path) == 0 for path in solution.paths):
        print(f"Error: Empty solution. Cannot create animation for {filename}")
        return

    anim = FuncAnimation(
        fig, 
        update, 
        frames=range(solution.makespan),
        init_func=init, 
        blit=False, 
        repeat=False
    )
    
    try:
        anim.save(filename, writer='pillow', fps=2)
        print(f"Visualization saved as '{filename}'")
    except Exception as e:
        print(f"Error saving animation '{filename}': {str(e)}")
    finally:
        plt.close(fig)

def create_solution_animation(
    env: Environment,
    solution: MAPFSolution, 
    filename: str = 'solution_animation.gif'
) -> None:
    fig, ax = plt.subplots(figsize=(10, 10))
    
    max_time = max(len(path) for path in solution.paths)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(solution.paths)))
    
    def animate(t):
        ax.clear()
        
        # Set up the plot
        ax.set_xlim(-0.5, env.size[1] - 0.5)
        ax.set_ylim(env.size[0] - 0.5, -0.5)
        ax.grid(True)
        
        # Plot obstacles
        for obs in env.obstacle_pos:
            ax.add_patch(plt.Rectangle((obs[1]-0.5, obs[0]-0.5), 1, 1, 
                                     facecolor='gray'))
        
        # Plot agents and their paths
        for agent_id, path in enumerate(solution.paths):
            # Plot the path up to current time
            if t > 0:
                positions = [(v.pos[1], v.pos[0]) for v in path[:t+1]]
                xs, ys = zip(*positions)
                ax.plot(xs, ys, '-', color=colors[agent_id], alpha=0.5)
            
            # Plot current position
            if t < len(path):
                pos = path[t].pos
                ax.plot(pos[1], pos[0], 'o', color=colors[agent_id], 
                       markersize=10)
        
        ax.set_title(f'Time step: {t}')
        return []
    
    anim = FuncAnimation(
        fig, 
        animate, 
        frames=max_time,
        interval=500,
        blit=True
    )
    
    anim.save(filename, writer='pillow', fps=2)
    plt.close()