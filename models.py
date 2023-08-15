from simulator import * 

def kronecker_pad(matrix, num_qubits, starting_site): 
    kron_list = [np.eye(2) for i in range(num_qubits)]    
    kron_list[starting_site] = matrix
    if matrix.shape[0] == 4: 
        del kron_list[starting_site+1]
    
    padded_matrix = kron_list[0]
    for i in range(1, len(kron_list)):
        padded_matrix = np.kron(kron_list[i], padded_matrix)    
    return padded_matrix

def XXZ(num_qubits, bias_coeff=1.0, hopping_coeff=1.0, unitary=False): 
    terms = []
    for i in range(num_qubits): 
        bias = bias_coeff*kronecker_pad(pauli[3], num_qubits, i)
        terms.append(bias)
        
    for i in range(num_qubits-1): 
        z_hop = hopping_coeff*kronecker_pad(pauli_tensor[(3,3)], num_qubits, i)
        terms.append(z_hop)
        y_hop = hopping_coeff*kronecker_pad(pauli_tensor[(2,2)], num_qubits, i)
        terms.append(y_hop)
        x_hop = hopping_coeff*kronecker_pad(pauli_tensor[(1,1)], num_qubits, i)
        terms.append(x_hop)
        
    if unitary: 
        return terms 
    else: 
        return sum(terms)
    
def trotter_XXZ(num_qubits, compression_step_size, trotter_steps, circuit=False): 
    ### this is not the best trotterization ### 
    gate_list = []
    for i in range(num_qubits): 
        bias = kronecker_pad(expm(1.j*pauli[3]*compression_step_size/trotter_steps), num_qubits, i)
        gate_list.append(bias)
        
    for i in range(num_qubits-1): 
        z_hop = kronecker_pad(expm(1.j*pauli_tensor[(3,3)]*compression_step_size/trotter_steps), num_qubits, i)
        gate_list.append(z_hop)
    for i in range(num_qubits-1):
        y_hop = kronecker_pad(expm(1.j*pauli_tensor[(2,2)]*compression_step_size/trotter_steps), num_qubits, i)
        gate_list.append(y_hop)
    for i in range(num_qubits-1):
        x_hop = kronecker_pad(expm(1.j*pauli_tensor[(1,1)]*compression_step_size/trotter_steps), num_qubits, i)
        gate_list.append(x_hop)
        
    if circuit: 
        return [gate_list] * trotter_steps
    else: 
        if trotter_steps == 1:
            return np.linalg.multi_dot(gate_list)
        else: 
            return np.linalg.multi_dot([np.linalg.multi_dot(gate_list)] * trotter_steps)