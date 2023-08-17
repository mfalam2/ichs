from simulator import *
from models import *
from tqdm import tqdm

@jit 
def all_partial_states(state_to_match, layer_list):
    ''' 
    takes overlap <0| L1^d L2^d |state_to_match> and breaks it down into pairs of left and right states: 
    |0>, L1^d L2^d |state_to_match>
    L1 |0>, L2^d |state_to_match>
    L2 L1 |0>, |state_to_match>
    '''    
    num_qubits = len(layer_list[0]) * 2
    ket = np.zeros(2**num_qubits)
    ket = ket.at[0].set(1.0)
    
    left_state_list = [ket]
    right_state_list = [state_to_match]
    num_layers = len(layer_list)
    for i in range(num_layers): 
        left_state_list.append(layer_action(left_state_list[-1], layer_list[i], parity=i%2, conj=False))
        right_state_list.insert(0, layer_action(right_state_list[0], layer_list[num_layers-1-i], parity=(num_layers-1-i)%2, conj=True))
    return left_state_list, right_state_list

@partial(jit, static_argnums=(3,4))
def eff_vector(left_state, right_state, layer, layer_idx, gate_idx):
    right_state = right_state.conj().T
    
    def deriv(i): 
        layer[gate_idx] = pauli_tensor[i%4][i//4]
        if layer_idx%2:
            return np.linalg.multi_dot([right_state, odd_layers(layer), left_state]).conj()
        else:
            return np.linalg.multi_dot([right_state, even_layers(layer), left_state]).conj()
    
    return np.array([deriv(i) for i in range(16)])
    
def sweep_up_down(left_state, right_state, layer, layer_idx, min_update=1e-4):
    cost_list = []
    converged = False
    while not converged: 
        for gate_idx in range(len(layer)): 
            eff_v = eff_vector(left_state, right_state, layer, layer_idx, gate_idx)
            env_mat = np.dot(eff_v.conj(), pauli_to_standard).reshape(4,4,order='F')
            u, s, vh = np.linalg.svd(env_mat)
            new_gate = vh.conj().T@u.conj().T  
            layer[gate_idx] = new_gate
        
        cost_list.append(1-sum(s)**2)
        if len(cost_list) > 2:
            if np.abs((cost_list[-1] - cost_list[-2])/cost_list[0]) < min_update: 
                converged = True        
    return layer, cost_list  

def sweep_left_right(state_to_match, ansatz, min_update=1e-6):
    left_state_list, right_state_list = all_partial_states(state_to_match, ansatz.layer_list)
    cost_list = [1 - np.abs(state_to_match.conj().T @ state_vector(ansatz.layer_list))**2]
    converged = False
    while not converged:
        for lidx in range(ansatz.num_layers): 
            new_layer, cost = sweep_up_down(left_state_list[lidx], right_state_list[lidx+1], ansatz.layer_list[lidx], lidx)
            ansatz.layer_list[lidx] = new_layer 
            left_state_list[lidx+1] = layer_action(left_state_list[lidx], new_layer, parity=lidx%2, conj=False)
        
        for lidx in range(ansatz.num_layers-1,-1,-1):
            new_layer, cost = sweep_up_down(left_state_list[lidx], right_state_list[lidx+1], ansatz.layer_list[lidx], lidx)
            ansatz.layer_list[lidx] = new_layer 
            right_state_list[lidx] = layer_action(right_state_list[lidx+1], new_layer, parity=lidx%2, conj=True)
        
        cost_list.append(cost[-1])
        if len(cost_list) > 2:
            if np.abs(cost_list[-1] - cost_list[-2]) < min_update: 
                converged = True  
    return ansatz, cost_list

def ichs(init_circ, total_time, compression_step_size, trotter_step_size, max_depth, min_update=1e-6): 
    ''' iteratively compressed hamiltonian simulation with the XXZ model '''
    cur_circ = copy.deepcopy(init_circ)
    exact_state = expm(1.j * total_time * XXZ(cur_circ.num_qubits)) @ state_vector(cur_circ.layer_list)
    trotter_op = trotter_XXZ(cur_circ.num_qubits, compression_step_size, int(compression_step_size/trotter_step_size))
    op_list = [trotter_op] * int(total_time/compression_step_size)
                                                   
    compression_infidelities = []
    for op in tqdm(op_list, desc=" circuit loop"):
        state_to_match = np.dot(op, state_vector(cur_circ.layer_list))
        ansatz = Ansatz(cur_circ.num_qubits, max_depth) if cur_circ.num_layers < max_depth else copy.deepcopy(cur_circ)
        cur_circ, cost_list = sweep_left_right(state_to_match, ansatz, min_update=min_update)
        compression_infidelities.append(cost_list)
    
    new_state = state_vector(cur_circ.layer_list)
    exact_fidelity = np.abs(new_state.conj().T @ exact_state)**2
    return cur_circ.layer_list, compression_infidelities, exact_fidelity