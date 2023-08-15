import copy
from functools import reduce, partial
import pickle  
import numpy as onp
import jax.numpy as np 
from jax.scipy.linalg import expm
from jax import jit, config
config.update("jax_enable_x64", True)

pickle_in = open('pauli_info.pickle', 'rb')
pauli, pauli_tensor, pauli_to_standard = pickle.load(pickle_in)
pickle_in.close()

#####################################
########## GATES ####################
#####################################
@jit
def one_qubit_gate(theta, lamb, phi): 
    ''' general single qubit gate from three Euler angles '''
    return np.array([[np.cos(theta/2), -np.exp(1.j*lamb)*np.sin(theta/2)], 
                 [np.exp(1.j*phi)*np.sin(theta/2), np.exp(1.j*(lamb+phi))*np.cos(theta/2)]])

@jit
def two_qubit_gate(params): 
    ''' general two qubit gate from 15 KAK parameters ''' 
    left_mat = np.einsum("ik,jl", one_qubit_gate(*params[0:3]), one_qubit_gate(*params[3:6])).reshape(4,4)
    arg = sum([params[i]*pauli_tensor[(i-5,i-5)] for i in range(6,9)])
    center_mat = expm(1.j*arg)
    right_mat = np.einsum("ik,jl", one_qubit_gate(*params[9:12]), one_qubit_gate(*params[12:15])).reshape(4,4)
    return np.einsum("ij,jk->ik", left_mat, np.einsum("lm,mn->ln", center_mat, right_mat))

####################################
########## LAYERS ##################
####################################
@jit
def even_layers(gates): 
    ''' compiles even layers of a brickwall circuit ''' 
    return reduce(np.kron, gates[::-1])

@jit
def odd_layers(gates): 
    ''' compiles odd layers of a brickwall circuit ''' 
    return np.kron(np.eye(2), np.kron(reduce(np.kron, gates[::-1]), np.eye(2)))

@partial(jit, static_argnums=(2,3))
def layer_action(cur_state, new_layer, parity, conj): 
    ''' acts with a layer of gates (even or odd) on a given state (bra or ket) ''' 
    layer_mat = odd_layers(new_layer) if parity else even_layers(new_layer)  
    return np.dot(layer_mat.conj().T, cur_state) if conj else np.dot(layer_mat, cur_state)

####################################
######### CIRCUITS #################
####################################
@jit 
def compile_circuit(layer_list, offset=0):
    ''' 
    compiles a circuit given as a list of layers of gates into a list of unitary matrices, 
    offset = 1 if first layer is short and not long 
    '''
    layer_mat_list = []
    for i in range(len(layer_list)): 
        if (i+offset)%2 == 0: 
            layer_mat_list.append(even_layers(layer_list[i]))
        else: 
            layer_mat_list.append(odd_layers(layer_list[i]))
    return layer_mat_list[::-1]
        
@jit
def state_vector(layer_list): 
    ''' acts with a circuit on the all-zero state ''' 
    num_qubits = len(layer_list[0])*2
    ket = np.zeros(2**num_qubits) 
    ket = ket.at[0].set(1.0)
    return np.linalg.multi_dot(compile_circuit(layer_list) + [ket]) 

class Ansatz:
    def __init__(self, num_qubits, num_layers, initial=0, KAK_list=None): 
        ''' initializes a parametrized brickwall ansatz ''' 
        self.num_qubits = num_qubits 
        self.num_layers = num_layers 
        self.num_gates = int((num_qubits*num_layers/2) - num_layers//2)
        
        self.KAK_list = KAK_list if KAK_list is not None else [2*onp.pi * onp.random.rand(15) for i in range(self.num_gates)]
        self.gate_list = [two_qubit_gate(params) for params in self.KAK_list]
        self.layer_list = self.gates_to_layers()      
   
    def gates_to_layers(self):
        ''' turns a list of gates into brickwall layer structure ''' 
        n = self.num_qubits
        return [self.gate_list[((i-1)//2)*(n-1)+(n//2):((i-1)//2)*(n-1)+(n//2)+(n//2)-1] 
                if i%2 else 
                self.gate_list[(i//2)*(n-1):(i//2)*(n-1)+(n//2)] for i in range(self.num_layers)]
