import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import colormaps
import matplotlib.animation as animation
from typing import Dict, List, Optional, Tuple, Any

from mapf_utils import Environment, MAPFSolution

class EnvironmentRenderer:
    def __init__(self):
        tab20b = colormaps.get_cmap('tab20b')
        colors = tab20b(list(range(20)))
        black = np.array([0,0,0,1])
        white = np.array([1,1,1,1])
        colors[0,:] = white  # Free space
        colors[1,:] = black  # Obstacles
        self.colors = colors
        self.cmap = ListedColormap(colors)
        
    def _get_color_idx(self, agent_id: int) -> int:
        return 2 + (agent_id % 18)  # Cycle through colors after first 2 (white/black)
        
    def draw_environment(self, 
                        ax: plt.Axes,
                        env: Environment,
                        agent_pos: Dict[int, Tuple[int, int]],
                        goals: Dict[int, Tuple[int, int]],
                        arrows: bool = True,
                        animated: bool = False,
                        grid: bool = False) -> Any:
        
        # Create matrix representation
        mat = env.dense_matrix()
        for agent_id in goals:
            mat[agent_pos[agent_id]] = self._get_color_idx(agent_id)
        
        # Draw base environment
        image = ax.imshow(mat, cmap=self.cmap, animated=True)

        if grid:
            # Major ticks
            ax.set_xticks(np.arange(0, mat.shape[1], 1))
            ax.set_yticks(np.arange(0, mat.shape[0], 1))

            # Minor ticks
            ax.set_xticks(np.arange(-.5, mat.shape[1], 1), minor=True)
            ax.set_yticks(np.arange(-.5, mat.shape[0], 1), minor=True)

            # Gridlines based on minor ticks
            ax.grid(which='minor', color='black', linestyle='-', linewidth=1)

            # Remove minor ticks
            ax.tick_params(which='minor', bottom=False, left=False)

        # Draw arrows pointing agents to their goals
        if arrows:
            for agent_id in goals:
                start_y, start_x = agent_pos[agent_id]
                end_y, end_x = goals[agent_id]
                dx = end_x - start_x
                dy = end_y - start_y
                ax.arrow(start_x, start_y, dx, dy, 
                        head_width=0.2, head_length=0.2, alpha=0.5)
        
        return image

class MAPFAnimator:
    def __init__(self, env: Environment, solution: MAPFSolution):
        self.renderer = EnvironmentRenderer()
        self.fig, self.ax = plt.subplots(figsize=(8,8))
        self.frames = []
        
        # Get final goals for reference
        self.final_goals = {
            agent_id: solution.paths[agent_id][-1].pos 
            for agent_id in solution.paths
        }
        
        # Create animation frames
        for t in range(solution.makespan):
            agent_positions = {}
            for agent_id in solution.paths:
                path = solution.paths[agent_id]
                if t < len(path):
                    agent_positions[agent_id] = path[t].pos
                else:
                    agent_positions[agent_id] = path[-1].pos
                    
            self.frames.append([
                self.renderer.draw_environment(
                    self.ax,
                    env,
                    agent_positions,
                    self.final_goals,
                    arrows=False,
                    animated=True
                )
            ])
            
            # Draw initial state
            if t == 0:
                self.renderer.draw_environment(
                    self.ax,
                    env,
                    agent_positions,
                    self.final_goals,
                    arrows=False,
                    animated=True
                )
                
    def save_animation(self, 
                      filename: str,
                      fps: int = 2,
                      repeat_delay: int = 5000) -> None:
        """Save the animation to a file"""
        anim = animation.ArtistAnimation(
            self.fig,
            self.frames,
            interval=1000/fps,  # Convert fps to interval
            repeat_delay=repeat_delay,
            repeat=True,
            blit=True
        )
        anim.save(filename)
        
    def show_animation(self,
                      interval: int = 500,
                      repeat_delay: int = 5000) -> animation.ArtistAnimation:
        """Display the animation in a notebook or interactive window"""
        return animation.ArtistAnimation(
            self.fig,
            self.frames,
            interval=interval,
            repeat_delay=repeat_delay,
            repeat=True,
            blit=True
        )
        
def draw_environment_state(env: Environment,
                         agent_positions: Dict[int, Tuple[int, int]],
                         goals: Dict[int, Tuple[int, int]],
                         figsize: Tuple[int, int] = (8, 8),
                         arrows: bool = True,
                         grid: bool = False) -> Tuple[plt.Figure, plt.Axes]:
    """Utility function to quickly visualize a single state"""
    fig, ax = plt.subplots(figsize=figsize)
    renderer = EnvironmentRenderer()
    renderer.draw_environment(ax, env, agent_positions, goals, arrows=arrows, grid=grid)
    return fig, ax