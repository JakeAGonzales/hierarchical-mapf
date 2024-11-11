import heapq
import copy
from typing import Optional, List, Tuple

from .constraints import PathVertex, PathEdge, ConstraintSet
from .problem import MAPFProblem, MAPFSolution
from .search import single_agent_astar

class CBSNode:
    def __init__(self):
        self.constraints = {}
        self.paths = []
        self.cost = 0

    def __lt__(self, other):
        return self.cost < other.cost

    def branch(self, agent1: int, agent2: int, constraint) -> Tuple['CBSNode', 'CBSNode']:
        left_node, right_node = CBSNode(), CBSNode()
        left_node.constraints = copy.deepcopy(self.constraints)
        right_node.constraints = copy.deepcopy(self.constraints)
        
        left_node.constraints.setdefault(agent1, ConstraintSet()).insert(constraint)
        right_node.constraints.setdefault(agent2, ConstraintSet()).insert(constraint)
        
        return left_node, right_node

def detect_conflicts(paths: List) -> List[Tuple[int, int, PathVertex]]:
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

def conflict_based_search(prob: MAPFProblem, time_limit: float = 60.0) -> Optional[MAPFSolution]:
    root = CBSNode()
    # Initialize root node
    for i, start in enumerate(prob.env.agent_pos):
        path_obj, _ = single_agent_astar(prob.env, start, prob.goals[i])
        if path_obj is None:
            return None
        root.paths.append(path_obj)
    
    root.cost = sum(path.cost for path in root.paths)
    queue = [root]
    seen_states = {}
    
    while queue:
        node = heapq.heappop(queue)
        conflicts = detect_conflicts(node.paths)
        
        if not conflicts:
            return MAPFSolution(node.paths)
            
        conflict = min(conflicts, key=lambda c: c[2].t if isinstance(c[2], PathVertex) else c[2].t)
        i, j, c = conflict
        
        for child in node.branch(i, j, c):
            child_valid = True
            new_paths = node.paths.copy()
            
            for agent in (i, j):
                path_obj, _ = single_agent_astar(
                    prob.env, 
                    prob.env.agent_pos[agent],
                    prob.goals[agent], 
                    child.constraints.get(agent)
                )
                if path_obj is None:
                    child_valid = False
                    break
                new_paths[agent] = path_obj
                
            if child_valid:
                child.paths = new_paths
                child.cost = sum(path.cost for path in child.paths)
                # Changed from vertexes to vertices
                child_hash = hash(tuple(tuple(path.vertices) for path in child.paths))
                if child_hash not in seen_states or child.cost < seen_states[child_hash]:
                    seen_states[child_hash] = child.cost
                    heapq.heappush(queue, child)
    
    return None