# cbs.py
import numpy as np
import networkx as nx
import torch
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import copy
import heapq
import time
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap as ListedColorMap
import random
import time
from collections import defaultdict


""""
This is an implementation of CBS that contains all the under the hood code
"""

class Action:
    NUM_ACTIONS = 5
    UP, DOWN, LEFT, RIGHT, WAIT = range(5)

class Environment:
    FREE, OBSTACLE, AGENT, GOAL, GOAL_REACHED = range(5)

    def __init__(self, size, obstacle_pos, agent_pos):
        self.size = size
        self.obstacle_pos = set(obstacle_pos)
        self.agent_pos = agent_pos
        self.agents = list(range(len(agent_pos)))
        self.graph = self._create_graph()
        self.data = {}
        for cartesian_index in obstacle_pos:
            self.data[cartesian_index] = self.OBSTACLE
        for i, cartesian_index in enumerate(agent_pos):
            self.data[cartesian_index] = self.AGENT

    def _create_graph(self):
        G = nx.grid_2d_graph(*self.size)
        for obs in self.obstacle_pos:
            G.remove_node(obs)
        return G

    def get_obstacles(self):
        return copy.deepcopy(self.obstacle_pos)
    
    def get_agents(self):
        return copy.deepcopy(self.agent_pos)

    def update_agent_pos(self, ids, positions):
        for i, j in enumerate(ids):
            if self.agent_pos[j] in self.data:
                del self.data[self.agent_pos[j]]
            self.data[positions[i]] = self.AGENT
            self.agent_pos[j] = positions[i]

    def dense_matrix(self):
        mat = np.zeros(self.size, dtype=int)
        for pos in self.data:
            mat[pos] = self.data[pos]
        return mat

def draw_environment(ax, env, goals, arrows=True, animated=False):
    mat = np.zeros(env.size, dtype=int)
    for loc in env.obstacle_pos:
        mat[loc] = env.OBSTACLE

    for id in env.agents:
        if env.agent_pos[id] == goals[id]:
            mat[goals[id]] = env.GOAL_REACHED
        else:
            mat[env.agent_pos[id]] = env.AGENT
    
    for id, pos in enumerate(goals):
        if mat[pos] != env.AGENT and mat[pos] != env.GOAL_REACHED:
            mat[pos] = env.GOAL

    colors = ["#FFFFFF", "#404040", "#0047AB", "#228B22", "#FFD700"]
    cmap = ListedColorMap(colors[0:np.max(mat)+1])
    image = ax.imshow(mat, cmap=cmap, animated=True)
    
    ax.set_xticks(np.arange(0, env.size[1], 1))
    ax.set_yticks(np.arange(0, env.size[0], 1))
    ax.set_xticklabels(range(1, env.size[1] + 1))
    ax.set_yticklabels(range(1, env.size[0] + 1))
    
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.tick_params(axis='both', which='major', length=0)
    
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    ax.grid(which='major', color='#E0E0E0', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    
    return [image]

def random_problem(size, n_agents: int, n_obstacles: int, seed=None):
    rng = np.random.default_rng(seed)
    all_pos = [(i, j) for i in range(size[0]) for j in range(size[1])]
    rng.shuffle(all_pos)
    
    agent_pos = all_pos[:n_agents]
    goal_pos = all_pos[n_agents:2*n_agents]
    obstacle_pos = all_pos[2*n_agents:2*n_agents+n_obstacles]

    env = Environment(size, obstacle_pos, agent_pos)
    return MAPFProblem(env, goal_pos)

class PathVertex:
    def __init__(self, pos, time):
        self.pos = pos
        self.t = time

    def __eq__(self, other):
        return isinstance(other, PathVertex) and self.pos == other.pos and self.t == other.t

    def __hash__(self):
        return hash((self.pos, self.t))

    def __lt__(self, other):
        return (self.t, self.pos) < (other.t, other.pos)

class Path:
    def __init__(self, vertices, cost):
        self.vertexes = vertices  
        self.cost = cost

    def __getitem__(self, i):
        return self.vertexes[i]

    def __len__(self):
        return len(self.vertexes)

class PathEdge:
    def __init__(self, p1: tuple, p2: tuple, t: int):
        self.p1 = p1
        self.p2 = p2
        self.t = t

    def compliment(self):
        return PathEdge(self.p2, self.p1, self.t)

    def __eq__(self, other):
        if not isinstance(other, PathEdge):
            return False
        return self.t == other.t and {self.p1, self.p2} == {other.p1, other.p2}

    def __hash__(self):
        return hash((frozenset([self.p1, self.p2]), self.t))

class ConstraintSet:
    def __init__(self, data = []):
        self._hashmap = {}
        for x in data:
            self._hashmap[x] = True

    def insert(self, x):
        self._hashmap[x] = True

    def __contains__(self, x):
        if isinstance(x, PathVertex):
            return x in self._hashmap
        elif isinstance(x, PathEdge):
            return x in self._hashmap or x.compliment() in self._hashmap
        return False

class CBSNode:
    def __init__(self):
        self.constraints = {}
        self.paths = []
        self.cost = 0

    def __lt__(self, other):
        return self.cost < other.cost

    def branch(self, agent1, agent2, constraint):
        left_node, right_node = CBSNode(), CBSNode()
        left_node.constraints = copy.deepcopy(self.constraints)
        right_node.constraints = copy.deepcopy(self.constraints)
        
        left_node.constraints.setdefault(agent1, ConstraintSet()).insert(constraint)
        right_node.constraints.setdefault(agent2, ConstraintSet()).insert(constraint)
        
        return left_node, right_node

class MAPFProblem:
    def __init__(self, environment, goals):
        if len(goals) != len(environment.agent_pos):
            raise ValueError("Goal states must match number of agents in environment")
        self.n_agents = len(goals)
        self.env = environment
        self.goals = goals
        

    def to_nparray(self, channel_padding = 0, agent_mask = []):
        N = len(self.goals) + channel_padding
        matshape = (*self.env.size, 2*N+1)
        mat = np.zeros(matshape, dtype=np.byte)
        for i, pos in enumerate(self.env.obstacle_pos):
            mat[*pos,0] = 1
        for i, pos in enumerate(self.env.agent_pos):
            if pos not in agent_mask:
                mat[*pos,i+1] = 1
        for i, pos in enumerate(self.goals):
            mat[*pos,(N+1)+i] = 1
        return mat

class MAPFSolution:
    def __init__(self, paths):
        self.paths = paths
        self.makespan = max(len(path) for path in paths)
    
    def __str__(self):
        n_agents = len(self.paths)
        mat = np.array([[f't = {i}' for i in range(self.makespan)]])
        for i in range(n_agents):
            arr = [f'{self.paths[i][t].pos}' if t < len(self.paths[i]) else None for t in range(self.makespan)]
            mat = np.vstack((mat, arr))
        return np.array_str(mat)
    
    def path_lengths(self):
        return [len(p) for p in self.paths]

def single_agent_astar(env, start, goal, constraints=None):
    def heuristic(a, b):
        return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5

    constraints = constraints or ConstraintSet()
    
    def is_valid(node, time, prev_node=None):
        vertex = PathVertex(node, time)
        if vertex in constraints:
            return False
        if prev_node is not None:
            edge = PathEdge(prev_node, node, time)
            if edge in constraints:
                return False
        return True

    open_set = [(0, start, 0, None)]  # (f_score, node, time, prev_node)
    came_from = {}
    g_score = {(start, 0): 0}
    f_score = {(start, 0): heuristic(start, goal)}
    goal_heuristic = {node: heuristic(node, goal) for node in env.graph.nodes()}

    while open_set:
        current_f, current_node, current_time, prev_node = heapq.heappop(open_set)

        if current_node == goal:
            path = []
            total_cost = g_score[(current_node, current_time)]
            while current_node:
                path.append(PathVertex(current_node, current_time))
                (current_node, current_time), prev_node = came_from.get((current_node, current_time), ((None, None), None))
            return Path(list(reversed(path)), total_cost), total_cost

        for neighbor in env.graph.neighbors(current_node):
            new_time = current_time + 1
            if not is_valid(neighbor, new_time, current_node):
                continue

            tentative_g_score = g_score[(current_node, current_time)] + 1

            if tentative_g_score < g_score.get((neighbor, new_time), float('inf')):
                came_from[(neighbor, new_time)] = ((current_node, current_time), prev_node)
                g_score[(neighbor, new_time)] = tentative_g_score
                f_score[(neighbor, new_time)] = tentative_g_score + goal_heuristic[neighbor]
                heapq.heappush(open_set, (f_score[(neighbor, new_time)], neighbor, new_time, current_node))

    print(f"No path found for agent from {start} to {goal}")
    return None, float('inf')

def detect_conflicts(paths):
    conflicts = []
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            path1, path2 = paths[i], paths[j]
            for t in range(min(len(path1), len(path2))):
                if path1[t].pos == path2[t].pos:
                    conflicts.append((i, j, PathVertex(path1[t].pos, t)))
                if t > 0 and path1[t].pos == path2[t-1].pos and path1[t-1].pos == path2[t].pos:
                    conflicts.append((i, j, PathEdge(path1[t-1].pos, path1[t].pos, t)))
    return conflicts

def conflict_based_search(prob, time_limit=60, warm_start_paths=None):
    root = CBSNode()
    
    if warm_start_paths:
        root.paths = convert_sampled_paths_to_cbs_format(warm_start_paths, prob)
    else:
        # Original initialization using A*
        for i, start in enumerate(prob.env.agent_pos):
            path, cost = single_agent_astar(prob.env, start, prob.goals[i])
            if path is None:
                return None
            root.paths.append(path)
    
    root.cost = sum(path.cost for path in root.paths)
    
    queue = [root]
    seen_states = {}  # Dictionary to store cost
    start_time = time.time()

    while queue:
        # Check if time limit has been exceeded
        if time.time() - start_time > time_limit:
            print(f"CBS search stopped: {time_limit} second time limit exceeded")
            return None
            
        node = heapq.heappop(queue)
        conflicts = detect_conflicts(node.paths)

        if not conflicts:
            elapsed_time = time.time() - start_time
            print(f"Solution found in {elapsed_time:.2f} seconds")
            return MAPFSolution(node.paths)  # Return optimal solution

        # Select the earliest conflict
        conflict = min(conflicts, key=lambda c: c[2].t if isinstance(c[2], PathVertex) else c[2].t)
        i, j, c = conflict
        
        for child in node.branch(i, j, c):
            child_valid = True
            new_paths = node.paths.copy()
            for agent in (i, j):
                path, cost = single_agent_astar(prob.env, prob.env.agent_pos[agent], prob.goals[agent], child.constraints.get(agent))
                if path is None:
                    child_valid = False
                    break
                new_paths[agent] = path

            if child_valid:
                child.paths = new_paths
                child.cost = sum(path.cost for path in child.paths)
                child_hash = hash(tuple(tuple(path.vertexes) for path in child.paths))
                
                if child_hash not in seen_states or child.cost < seen_states[child_hash]:
                    seen_states[child_hash] = child.cost
                    heapq.heappush(queue, child)

    return None  # No solution found within time limit

def generate_heatmap(predicted_flows, grid_size):
    if isinstance(predicted_flows, torch.Tensor):
        flow = predicted_flows[:, 2].cpu().numpy().reshape(grid_size, grid_size)
    else:
        flow = np.array(predicted_flows)[:, 2].reshape(grid_size, grid_size)
    return flow

def create_custom_colormap():
    colors = ['white', '#E6F3FF', '#B3DBFF', '#80C3FF', '#4DA6FF', '#1A8CFF', '#0066CC']
    n_bins = 100  # Number of color gradations
    return LinearSegmentedColormap.from_list("custom_blue", colors, N=n_bins)

def overlay_heatmap(ax, heatmap, alpha=0.6):
    cmap = create_custom_colormap()
    normalized_heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
    threshold = 0.05  
    normalized_heatmap[normalized_heatmap < threshold] = 0
    normalized_heatmap = np.power(normalized_heatmap, 0.7)  
    im = ax.imshow(normalized_heatmap, cmap=cmap, alpha=alpha, vmin=0, vmax=1)
    return im

class MAPFAnimation:
    def __init__(self, prob, solution):
        self.prob = prob
        self.solution = solution
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.frames = []
        
        for t in range(solution.makespan):
            positions = [path[min(t, len(path)-1)].pos for path in solution.paths]
            env_copy = copy.deepcopy(prob.env)
            env_copy.update_agent_pos(range(len(positions)), positions)
            self.frames.append((env_copy, prob.goals))

        # Set up the axis limits
        self.ax.set_xlim(-0.5, prob.env.size[1] - 0.5)
        self.ax.set_ylim(prob.env.size[0] - 0.5, -0.5)

    def draw_path_traces(self, ax, frame):
        for path in self.solution.paths:
            # Draw up to the current frame
            path_to_draw = path[:frame+1]
            xs, ys = zip(*[v.pos for v in path_to_draw])
            ax.plot(ys, xs, color='#DC143C', linewidth=2, alpha=0.5)
            # Add a marker at the current position
            ax.plot(ys[-1], xs[-1], 'o', color='#DC143C', markersize=8)

    def animate(self, frame):
        self.ax.clear()  
        env, goals = self.frames[frame]
        images = draw_environment(self.ax, env, goals, arrows=False, animated=True)
        
        self.draw_path_traces(self.ax, frame)
        
        self.ax.text(0.02, 0.98, f'Time step: {frame}', transform=self.ax.transAxes, 
                     verticalalignment='top', fontsize=10)
        
        self.ax.set_xticks(range(self.prob.env.size[1]))
        self.ax.set_yticks(range(self.prob.env.size[0]))
        self.ax.set_xticklabels(range(1, self.prob.env.size[1] + 1))
        self.ax.set_yticklabels(range(1, self.prob.env.size[0] + 1))
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        
        return images

    def save(self, filename, fps=2):
        anim = FuncAnimation(self.fig, self.animate, frames=len(self.frames), interval=500, blit=False, repeat=True)
        anim.save(filename, writer='pillow', fps=fps, dpi=300)
        plt.close(self.fig)

# NEW CODE

def convert_sampled_paths_to_cbs_format(sampled_paths, prob):
    """
    Convert sampled paths to CBS format.
    sampled_paths: Dictionary of agent_id -> list of PathVertex objects
    """
    cbs_paths = []
    for agent_id, path in sampled_paths.items():
        if not path:  
            continue

        if path[-1].pos != prob.goals[agent_id]:
            extend_path_to_goal(path, prob.goals[agent_id], prob.env)
        
        cbs_paths.append(Path(path, len(path)))
    
    return cbs_paths

def extend_path_to_goal(path, goal, env):
    """
    Extend a path of PathVertex objects to reach the goal
    """
    last_pos = path[-1].pos
    last_time = path[-1].t
    
    while last_pos != goal:
        next_pos = min(env.graph.neighbors(last_pos), 
                      key=lambda x: manhattan_distance(x, goal))
        last_time += 1
        path.append(PathVertex(next_pos, last_time))
        last_pos = next_pos

def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def cbs_with_warm_start(prob, sampled_paths, max_iterations=1000):
    warm_start_time = time.time()
    solution = conflict_based_search(prob, max_iterations, warm_start_paths=sampled_paths)
    warm_start_time = time.time() - warm_start_time

    if solution:
        print(f"CBS with warm start found a solution in {warm_start_time:.2f} seconds")
        print(f"Makespan: {solution.makespan}")
        print(f"Sum of costs: {sum(len(path) for path in solution.paths)}")
    else:
        print("CBS with warm start couldn't find a solution")

    return solution, warm_start_time