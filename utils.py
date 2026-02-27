
import networkx as nx
import numpy as np
from scipy.optimize import fsolve
from scipy.stats import beta
import torch.nn as nn
import torch

def shortest_path_with_k_nodes_neg(graph, s, t, k):
    """
    Compute the shortest path from s to t that passes through exactly k vertices.
    Supports negative edge weights, but not negative cycles.
    """
    nodes = list(graph.nodes)
    n = len(nodes)

    # init dp
    dp = [[float('inf')] * (k + 1) for _ in range(n)]
    node_to_idx = {v: i for i, v in enumerate(nodes)}

    start_idx = node_to_idx[s]
    dp[start_idx][1] = 0


    prev = [[None] * (k + 1) for _ in range(n)]

    # Bellman-Ford style relax.
    for step in range(2, k + 1):
        for v in nodes:
            v_idx = node_to_idx[v]
            for u in graph.predecessors(v):
                u_idx = node_to_idx[u]
                weight = graph[u][v]['weight']

                if dp[u_idx][step-1] != float('inf') and dp[u_idx][step-1] + weight < dp[v_idx][step]:
                    dp[v_idx][step] = dp[u_idx][step-1] + weight
                    prev[v_idx][step] = u


    target_idx = node_to_idx[t]
    if dp[target_idx][k] == float('inf'):
        return None, None

    path = []
    curr = t
    step = k
    while curr is not None and step > 0:
        path.append(curr)
        curr_idx = node_to_idx[curr]
        curr = prev[curr_idx][step]
        step -= 1
    path.reverse()

    return dp[target_idx][k], path

def solve_adaptive_graph_energy_convergence(graph, s, t, k_max=50, fidelity_target=0.99):
    """
    Energy Convergence Based Planner (System 2 Decision Logic)

    Parameters:
        - fidelity_target: Energy capture threshold. For example, 0.99 indicates stopping when the sampled path captures 99% of the total denoising gain.
    """
    nodes = list(graph.nodes)
    n = len(nodes)
    node_to_idx = {v: i for i, v in enumerate(nodes)}
    target_idx = node_to_idx[t]
    start_idx = node_to_idx[s]


    dp = [[float('inf')] * (k_max + 1) for _ in range(n)]
    prev = [[None] * (k_max + 1) for _ in range(n)]
    dp[start_idx][1] = 0

    for step in range(2, k_max + 1):
        for v in nodes:
            v_idx = node_to_idx[v]
            for u in graph.predecessors(v):
                u_idx = node_to_idx[u]
                if dp[u_idx][step-1] != float('inf'):
                    new_cost = dp[u_idx][step-1] + graph[u][v]['weight']
                    if new_cost < dp[v_idx][step]:
                        dp[v_idx][step] = new_cost
                        prev[v_idx][step] = u


    costs = dp[target_idx]
    valid_ks = [k for k in range(1, k_max + 1) if costs[k] != float('inf')]
    
    if len(valid_ks) < 2:
        return None,[]


    total_potential_gain = costs[valid_ks[0]] - costs[valid_ks[-1]]
    
    best_k = valid_ks[-1]
    
    for k in valid_ks:
        current_gain = costs[valid_ks[0]] - costs[k]
        
        explained_gain_ratio = current_gain / (total_potential_gain + 1e-12)

        if explained_gain_ratio >= fidelity_target:
            best_k = k
            break
        
    
    path = []
    curr = t
    s_idx = best_k
    while curr is not None and s_idx > 0:
        path.append(curr)
        curr = prev[node_to_idx[curr]][s_idx]
        s_idx -= 1
    path.reverse()
    
    return costs[best_k], path    
    

class GraphSearch():
    
    def __init__(self, dna_list, super_k=None, np_timesteps=None):
        self.T = len(dna_list)
        if np_timesteps is not None:
            assert len(dna_list) == np_timesteps.shape[0]
            t_np_list = np_timesteps
        else:
            t_np_list = np.linspace(1 / self.T, 1, self.T)
        
        
        
        self.G = nx.DiGraph()
        self.G.add_nodes_from(range(self.T+2))

        for i in range(self.T):
            for j in range(i+1, self.T):
                w  = dna_list[j] * ((t_np_list[j]-t_np_list[i])/t_np_list[j])**2
                self.G.add_edge(i, j, weight=w)
        
        super_k = self.T if super_k is None else super_k
        self.super_k = super_k
        if super_k==0:
            self.begin, self.end = 0, self.T-1
        else:
            self.begin, self.end = self.T, self.T+1

        for i in range(super_k):
            self.G.add_edge(self.T-i-1, self.end, weight = -dna_list[self.T-i-1])
            self.G.add_edge(self.begin, i, weight = dna_list[i])
                
        self.t_np_list = t_np_list
        
        
    def path2times(self, path):
        # if path[0]>=self.T:
        #     path = path[1:]
        # if path[-1]>=self.T:
        #     path=path[:-1]
       
        # path[-1]=self.T-1
        # path[-1] = self.T-1
        path = path[::-1]
        times_optimal = [self.t_np_list[v] for v in path]
        return path, times_optimal
    
    def get_adj_matrix(self):
        adj_matrix = nx.to_numpy_array(self.G, nodelist=self.G.nodes, weight='weight')
        return adj_matrix
    
    
    def find_optimal_k_times(self, k):
        k = k if self.super_k==0 else k+2
        cost, path = shortest_path_with_k_nodes_neg(self.G, self.begin, self.end, k)
        path = path if self.super_k==0 else path[1:-1]
        path, times_optimal = self.path2times(path)
        return cost, path, times_optimal
    
    def find_optimal_adaptive_times(self, inference_steps_max = 50, fidelity_target=0.99):
        inference_steps_max = inference_steps_max if self.super_k==0 else inference_steps_max+2
        cost, path = solve_adaptive_graph_energy_convergence(self.G, 
                                               self.begin,
                                               self.end, 
                                               k_max=inference_steps_max,
                                               fidelity_target  = fidelity_target)
                                               
        path = path if self.super_k==0 else path[1:-1]
        path, times_optimal = self.path2times(path)
        return cost, path, times_optimal
    




class SimpleMLP(nn.Module):
    def __init__(self, input_dim=3584, output_dim=100, hidden_dim=256,dropout_raio=0.6):
        super(SimpleMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_raio), 
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout_raio),
            nn.Linear(hidden_dim//2, output_dim), 
        )
        
    def forward(self, x):
        return self.net(x)



def inverse_euler_set_timesteps(final_sigmas, scheduler_config, mu):
    """
    Inverse function to recover the original sigmas before set_timesteps transforms.

    Args:
        final_sigmas (torch.Tensor or np.ndarray): The final sigmas returned by set_timesteps
        scheduler_config (dict): Configuration of the scheduler
        mu (float): The mu value used in dynamic shifting

    Returns:
        np.ndarray: The original sigma values before shifting/stretching/conversion
    """
    final_sigmas = final_sigmas.cpu().numpy() if torch.is_tensor(final_sigmas) else np.array(final_sigmas)
    
    # Remove appended terminal sigma (0 or 1)
    if final_sigmas[-1]==0.0:
        final_sigmas = final_sigmas[:-1]
    
    # Step 1: Undo inversion if invert_sigmas was True
    if scheduler_config["invert_sigmas"]:
        #print('invert_sigmas')
        final_sigmas = 1.0 - final_sigmas
    
    # Step 2: Undo Karras/Exponential/Beta conversion if applied
    num_steps = len(final_sigmas)
    sigma_min = final_sigmas[-1]
    sigma_max = final_sigmas[0]

    if scheduler_config["use_karras_sigmas"]:
        #print('use_karras_sigmas')
        rho = 7.0
        ramp = np.linspace(0, 1, num_steps)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        # Solve for original sigmas
        original_sigmas = ( (final_sigmas ** (1/rho) - max_inv_rho) / (min_inv_rho - max_inv_rho) )
        original_sigmas = sigma_max + (sigma_min - sigma_max) * original_sigmas
    elif scheduler_config["use_exponential_sigmas"]:
        #print('use_exponential_sigmas')
        original_sigmas = np.exp(np.linspace(np.log(sigma_max), np.log(sigma_min), num_steps))
        # Note: approximate, because exponential mapping is monotonic
    elif scheduler_config["use_beta_sigmas"]:
        #print('use_beta_sigmas')
        # Solve numerically for each sigma
        alpha = 0.6
        beta_param = 0.6
        timesteps = []
        for s in final_sigmas:
            def f(t):
                return sigma_min + (sigma_max - sigma_min) * beta.ppf(1 - t, alpha, beta_param) - s
            t0 = fsolve(f, 0.5)
            timesteps.append(t0[0])
        timesteps = np.array(timesteps)
        original_sigmas = timesteps / scheduler_config["num_train_timesteps"]
    else:
        original_sigmas = final_sigmas.copy()
    
    # Step 3: Undo stretch_shift_to_terminal if shift_terminal is set
    if scheduler_config["shift_terminal"] is not None:
        #print('use shift_terminal')
        src_last_element = original_sigmas.shape[0]/scheduler_config["num_train_timesteps"]
        
        one_minus_z_last = 1 - src_last_element
        epsilon = 1e-9 # 避免除以零
        scale_factor = one_minus_z_last / (1 - scheduler_config["shift_terminal"] + epsilon)
        one_minus_z = (1 - original_sigmas) * scale_factor
        original_sigmas = 1 - one_minus_z
        # print('use shift_terminal')
        # one_minus_z = 1 - original_sigmas
        # scale_factor = one_minus_z[-1] / (1 - scheduler_config["shift_terminal"])
        # original_sigmas = 1 - (one_minus_z * scale_factor)
        
        # one_minus_z = 1 - t
        # scale_factor = one_minus_z[-1] / (1 - self.config.shift_terminal)
        # stretched_t = 1 - (one_minus_z / scale_factor)
        # return stretched_t
    # Step 4: Undo dynamic shifting if used
    if scheduler_config["use_dynamic_shifting"]:
        #print('use_dynamic_shifting')
        if scheduler_config["time_shift_type"] == "exponential":
            # Solve exp(mu) / (exp(mu) + (1/t - 1)^sigma) = final_sigma for t
            def invert_exponential(fs):
                def func(t):
                    return np.exp(mu) / (np.exp(mu) + (1/t - 1)) - fs
                t0 = fsolve(func, 0.5)
                return t0[0]
            original_sigmas = np.array([invert_exponential(s) for s in original_sigmas])
        elif scheduler_config["time_shift_type"] == "linear":
            def invert_linear(fs):
                def func(t):
                    return mu / (mu + (1/t - 1)) - fs
                t0 = fsolve(func, 0.5)
                return t0[0]
            original_sigmas = np.array([invert_linear(s) for s in original_sigmas])
        print(scheduler_config["time_shift_type"] )
    else:
        # Undo static shift
        shift = scheduler_config["shift"]
        print('shift')
        original_sigmas = original_sigmas / (shift - (shift - 1) * original_sigmas)
    
    return original_sigmas