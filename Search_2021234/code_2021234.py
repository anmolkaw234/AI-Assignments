import numpy as np
import pickle
import heapq
from collections import deque
import math
import unittest
from itertools import count

# General Notes:
# - Update the provided file name (code_<RollNumber>.py) as per the instructions.
# - Do not change the function name, number of parameters or the sequence of parameters.
# - The expected output for each function is a path (list of node names)
# - Ensure that the returned path includes both the start node and the goal node, in the correct order.
# - If no valid path exists between the start and goal nodes, the function should return None.


# Algorithm: Iterative Deepening Search (IDS)

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 2, 9, 8, 5, 97, 98, 12]

class IDS:
    def __init__(self, adj_matrix) -> None:
        self.adj_matrix = adj_matrix
    
    def maximum_depth_dfs(self, node, viewed, depth_current_value):
        viewed.add(node)
        maximum_depth_value = depth_current_value
        
        neighbor = 0
        while neighbor < len(self.adj_matrix[node]):
            joined = self.adj_matrix[node][neighbor]
            if joined > 0 and neighbor not in viewed:
                id_depth = self.maximum_depth_dfs(neighbor, viewed, depth_current_value + 1)
                maximum_depth_value = max(maximum_depth_value, id_depth)
            neighbor += 1
        return maximum_depth_value
    
    def maximum_depth_value(self, start_node):
        viewed = set()
        return self.maximum_depth_dfs(start_node, viewed, 0)
    
    def depth_limited_search(self, node, goal, depth_max, path, viewed):
        if node == goal:
            return path
        
        if depth_max <= 0:
            return None
        
        viewed.add(node)
        neighbor = 0
        while neighbor < len(self.adj_matrix[node]):
            joined = self.adj_matrix[node][neighbor]
            if neighbor in viewed or joined < 1:
                neighbor += 1
                continue
            result = self.depth_limited_search(neighbor, goal, depth_max-1, path + [neighbor], viewed)
            if result:
                return result
            neighbor += 1
            
        viewed.remove(node)
        return None

    def search_ids(self, source, destination):
        maximum_depth = self.maximum_depth_value(source)
        for depth in range(maximum_depth):
            viewed = set()
            result = self.depth_limited_search(source, destination, depth, [source], viewed)
            if result:
                return result
        return None
    
def get_ids_path(adj_matrix, start_node, goal_node):
    ids = IDS(adj_matrix)
    return ids.search_ids(start_node, goal_node)




# Algorithm: Bi-Directional Search

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 2, 9, 8, 5, 97, 98, 12]



def get_bidirectional_search_path(adj_matrix, start_node, goal_node):


    node_name = set()
    n = adj_matrix.shape[0]
    i = 0
    while i < n:
        node_name.add(i)
        j = 0
        while j < n:
            if adj_matrix[i][j] > 0:
                node_name.update([i, j])
            j += 1
        i += 1


    node_name.update([start_node, goal_node])
    node_name = sorted(node_name)


    path_node_index = {label: idx for idx, label in enumerate(node_name)}
    path_index_node = {idx: label for label, idx in path_node_index.items()}

    number_of_nodes = len(node_name)

    new_adjacency_matrix = np.zeros((number_of_nodes, number_of_nodes), dtype=int)
    i = 0
    while i < n:
        j = 0
        while j < n:
            if adj_matrix[i][j] > 0:
                idx_i = path_node_index.get(i)
                idx_j = path_node_index.get(j)
                if idx_i is not None and idx_j is not None:
                    new_adjacency_matrix[idx_i][idx_j] = adj_matrix[i][j]
            j += 1
        i += 1
    adj_matrix = new_adjacency_matrix

    start_index = path_node_index[start_node]
    goal_index = path_node_index[goal_node]

    if start_index == goal_index:
        return [start_node]

    viewed_from_start = {start_index}
    viewed_from_goal = {goal_index}

    parent_from_start = {start_index: None}
    parent_from_goal = {goal_index: None}

    queue_start = deque([start_index])
    queue_goal = deque([goal_index])

    meeting_node = None

    for _ in iter(int, 1):
        if not queue_start or not queue_goal:
            break

        current_level_size = len(queue_start)
        for _ in range(current_level_size):
            node = queue_start.popleft()
            neighbors = [i for i, val in enumerate(adj_matrix[node]) if val > 0]
            idx = 0
            while idx < len(neighbors):
                neighbor = neighbors[idx]
                if neighbor not in viewed_from_start:
                    viewed_from_start.add(neighbor)
                    parent_from_start[neighbor] = node
                    queue_start.append(neighbor)
                    if neighbor in viewed_from_goal:
                        meeting_node = neighbor
                        break
                idx += 1
            if meeting_node is not None:
                break
        if meeting_node is not None:
            break

        current_level_size = len(queue_goal)
        for _ in range(current_level_size):
            node = queue_goal.popleft()
            neighbors = [i for i, val in enumerate(adj_matrix[node]) if val > 0]
            idx = 0
            while idx < len(neighbors):
                neighbor = neighbors[idx]
                if neighbor not in viewed_from_goal:
                    viewed_from_goal.add(neighbor)
                    parent_from_goal[neighbor] = node
                    queue_goal.append(neighbor)
                    if neighbor in viewed_from_start:
                        meeting_node = neighbor
                        break
                idx += 1
            if meeting_node is not None:
                break

    if meeting_node is None:
        return None

    path_start = []
    node = meeting_node
    while node is not None:
        path_start.append(path_index_node[node])
        node = parent_from_start[node]
    path_start.reverse()

    path_goal = []
    node = parent_from_goal[meeting_node]
    while node is not None:
        path_goal.append(path_index_node[node])
        node = parent_from_goal[node]

    full_path = path_start + path_goal

    return full_path






# Algorithm: A* Search Algorithm

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - node_attributes: Dictionary of node attributes containing x, y coordinates for heuristic calculations.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 27, 9, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 28, 10, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 27, 9, 8, 5, 97, 28, 10, 12]


# A* Search Algorithm
def dist(u, v, attr):
    x1, y1 = attr[u]['x'], attr[u]['y']
    x2, y2 = attr[v]['x'], attr[v]['y']
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

class Astr:
    def __init__(self, adj_matrix, node_attributes) -> None:
        self.adj_matrix = adj_matrix
        self.node_attributes = node_attributes
        self.n = len(adj_matrix)
        
    def reconstruct_path(self, parent, start_node, goal_node):
        path = []
        recent = goal_node
        for _ in iter(int, 1):
            if recent is None:
                break
            path.append(recent)
            recent = parent[recent]
        path.reverse()
        return path
    
    def astr_search(self, start_node, goal_node):
        open_list = []
        
        g_scores = {i: float('inf') for i in range(self.n)}
        g_scores[start_node] = 0
        
        f_scores = {i: float('inf') for i in range(self.n)}
        f_scores[start_node] = dist(start_node, goal_node, self.node_attributes)

        heapq.heappush(open_list, (f_scores[start_node], start_node))
        parent = {start_node: None}
        closed_list = set()

        for _ in iter(int, 1):
            if not open_list:
                break
            current_f, current_node = heapq.heappop(open_list)
            if current_node == goal_node:
                return self.reconstruct_path(parent, start_node, goal_node)

            closed_list.add(current_node)
            v = 0
            while v < self.n:
                w = self.adj_matrix[current_node][v]
                if w == 0 or v in closed_list:
                    v += 1
                    continue

                initial_gscore = g_scores[current_node] + w

                if initial_gscore < g_scores[v]:
                    parent[v] = current_node
                    g_scores[v] = initial_gscore
                    h_score = dist(v, goal_node, self.node_attributes)
                    f_scores[v] = initial_gscore + h_score

                    heapq.heappush(open_list, (f_scores[v], v))
                v += 1

        return None

def get_astar_search_path(adj_matrix, node_attributes, start_node, goal_node):
    astar = Astr(adj_matrix, node_attributes)
    return astar.astr_search(start_node, goal_node)



# Algorithm: Bi-Directional Heuristic Search

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - node_attributes: Dictionary of node attributes containing x, y coordinates for heuristic calculations.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 27, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 34, 33, 11, 32, 31, 3, 5, 97, 28, 10, 12]




def calculate_distance(node_a, node_b, attributes):
    x1, y1 = attributes[node_a]['x'], attributes[node_a]['y']
    x2, y2 = attributes[node_b]['x'], attributes[node_b]['y']
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def rebuild_path(start, goal, predecessors):
    path = []
    while goal is not None:
        path.append(goal)
        goal = predecessors[goal]
    return path[::-1]

class BiDirectionalHeuristicSearch:
    def __init__(self, num_nodes, adjacency_matrix, node_attrs):
        self.num_nodes = num_nodes
        self.adj_matrix = adjacency_matrix
        self.node_attrs = node_attrs
        
        self.forward_queue = []
        self.backward_queue = []
        
        self.forward_visited = [False] * num_nodes
        self.backward_visited = [False] * num_nodes
        
        
        self.forward_parents = [None] * num_nodes
        self.backward_parents = [None] * num_nodes
        
    
        self.forward_costs = [float('inf')] * num_nodes
        self.backward_costs = [float('inf')] * num_nodes
        
    def forward_search_step(self, goal):
        current_cost, current_node = heapq.heappop(self.forward_queue)
        for neighbor, edge_cost in enumerate(self.adj_matrix[current_node]):
            if self.forward_visited[neighbor] or edge_cost < 1:
                continue
            new_cost = self.forward_costs[current_node] + edge_cost
            if new_cost < self.forward_costs[neighbor]:
                self.forward_visited[neighbor] = True
                self.forward_parents[neighbor] = current_node
                self.forward_costs[neighbor] = new_cost
                heuristic = calculate_distance(neighbor, goal, self.node_attrs)
                heapq.heappush(self.forward_queue, (new_cost + heuristic, neighbor))
    
    def backward_search_step(self, start):
        current_cost, current_node = heapq.heappop(self.backward_queue)
        for neighbor, edge_cost in enumerate(self.adj_matrix[current_node]):
            if self.backward_visited[neighbor] or edge_cost < 1:
                continue
            new_cost = self.backward_costs[current_node]
            if new_cost < self.backward_costs[neighbor]:
                self.backward_visited[neighbor] = True
                self.backward_parents[neighbor] = current_node
                self.backward_costs[neighbor] = new_cost
                heuristic = calculate_distance(neighbor, start, self.node_attrs)
                heapq.heappush(self.backward_queue, (new_cost + heuristic, neighbor))
    
    def find_intersection(self):
        best_node = -1
        minimal_cost = float('inf')
        for node in range(self.num_nodes):
            if self.forward_visited[node] and self.backward_visited[node]:
                total_cost = self.forward_costs[node] + self.backward_costs[node]
                if total_cost < minimal_cost:
                    minimal_cost = total_cost
                    best_node = node
        return best_node
  
    def construct_complete_path(self, intersect_node, start, goal):
        path = []
        node = intersect_node
        while node != start:
            path.append(node)
            node = self.forward_parents[node]
        path.append(start)
        path.reverse()
        
        node = intersect_node
        while node != goal:
            path.append(self.backward_parents[node])
            node = self.backward_parents[node]
        
        return path
      
    def execute_search(self, start, goal):
        heapq.heappush(self.forward_queue, (calculate_distance(start, goal, self.node_attrs), start))
        self.forward_visited[start] = True
        self.forward_costs[start] = 0
        self.forward_parents[start] = -1
        
        heapq.heappush(self.backward_queue, (calculate_distance(goal, start, self.node_attrs), goal))
        self.backward_visited[goal] = True
        self.backward_costs[goal] = 0
        self.backward_parents[goal] = -1
  
        while self.forward_queue and self.backward_queue:
            self.forward_search_step(goal)
            self.backward_search_step(start)
            
            intersection = self.find_intersection()
            if intersection != -1:
                return self.construct_complete_path(intersection, start, goal)
            
def get_bidirectional_heuristic_search_path(adj_matrix, node_attributes, start_node, goal_node):
    n = len(adj_matrix)
    astar_bid = BiDirectionalHeuristicSearch(num_nodes=n, adjacency_matrix=adj_matrix, node_attrs=node_attributes)
    return astar_bid.execute_search(start=start_node, goal=goal_node)




# Bonus Problem
 
# Input:
# - adj_matrix: A 2D list or numpy array representing the adjacency matrix of the graph.

# Return:
# - A list of tuples where each tuple (u, v) represents an edge between nodes u and v.
#   These are the vulnerable roads whose removal would disconnect parts of the graph.

# Note:
# - The graph is undirected, so if an edge (u, v) is vulnerable, then (v, u) should not be repeated in the output list.
# - If the input graph has no vulnerable roads, return an empty list [].

def bonus_problem(adj_matrix):
    def create_adj_list(mat):
        adj_list = {idx: [] for idx in range(len(mat))}
        idx = 0
        while idx < len(mat):
            jdx = 0
            while jdx < len(mat):
                if mat[idx][jdx] > 0:
                    adj_list[idx].append(jdx)
                jdx += 1
            idx += 1
        return adj_list
    
    def dfs(node, parent, disc, low, curr_time, bridge_list, adj_list, viewed_nodes):
        viewed_nodes[node] = True
        disc[node] = low[node] = curr_time[0]
        curr_time[0] += 1
        
        nbr = 0
        while nbr < len(adj_list[node]):
            neighbor = adj_list[node][nbr]
            nbr += 1
            if neighbor == parent:
                continue
            if not viewed_nodes[neighbor]:
                dfs(neighbor, node, disc, low, curr_time, bridge_list, adj_list, viewed_nodes)
                low[node] = min(low[node], low[neighbor])
                if low[neighbor] > disc[node]:
                    bridge_list.append((node, neighbor))
            else:
                low[node] = min(low[node], disc[neighbor])
                
    adj_list = create_adj_list(adj_matrix)
    num_nodes = len(adj_matrix)
    discovery_time = [-1] * num_nodes
    low_point = [-1] * num_nodes
    viewed = [False] * num_nodes
    bridges = []
    curr_time = [0]
    
    idx = 0
    while idx < num_nodes:
        if not viewed[idx]:
            dfs(idx, -1, discovery_time, low_point, curr_time, bridges, adj_list, viewed)
        idx += 1
        
    return bridges




if __name__ == "__main__":
  adj_matrix = np.load('IIIT_Delhi.npy')
  with open('IIIT_Delhi.pkl', 'rb') as f:
    node_attributes = pickle.load(f)

  start_node = int(input("Enter the start node: "))
  end_node = int(input("Enter the end node: "))

  print(f'Iterative Deepening Search Path: {get_ids_path(adj_matrix,start_node,end_node)}')
  print(f'Bidirectional Search Path: {get_bidirectional_search_path(adj_matrix,start_node,end_node)}')
  print(f'A* Path: {get_astar_search_path(adj_matrix,node_attributes,start_node,end_node)}')
  print(f'Bidirectional Heuristic Search Path: {get_bidirectional_heuristic_search_path(adj_matrix,node_attributes,start_node,end_node)}')
  print(f'Bonus Problem: {bonus_problem(adj_matrix)}')