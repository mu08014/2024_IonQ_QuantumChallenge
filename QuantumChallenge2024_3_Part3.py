import numpy as np
from scipy.linalg import eigh, expm
from scipy.optimize import minimize
from scipy.spatial.distance import euclidean
from itertools import combinations
import matplotlib.pyplot as plt
from qiskit import *
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram

X = np.array([[0., 1.], [1., 0.]])
Y = np.array([[0., -1j], [1j, 0.]])
Z = np.array([[1., 0.], [0., -1.]])
I = np.array([[1., 0.], [0., 1.]])

H_0_1_XX = np.kron(I, np.kron(I, np.kron(X, X)))
H_0_1_YY = np.kron(I, np.kron(I, np.kron(Y, Y)))
H_0_1_ZZ = np.kron(I, np.kron(I, np.kron(Z, Z)))
H_0_2_XX = np.kron(I, np.kron(X, np.kron(I, X)))
H_0_2_YY = np.kron(I, np.kron(Y, np.kron(I, Y)))
H_0_2_ZZ = np.kron(I, np.kron(Z, np.kron(I, Z)))
H_0_3_XX = np.kron(X, np.kron(I, np.kron(I, X)))
H_0_3_YY = np.kron(Y, np.kron(I, np.kron(I, Y)))
H_0_3_ZZ = np.kron(Z, np.kron(I, np.kron(I, Z)))
H_1_2_XX = np.kron(I, np.kron(X, np.kron(X, I)))
H_1_2_YY = np.kron(I, np.kron(Y, np.kron(Y, I)))
H_1_2_ZZ = np.kron(I, np.kron(Z, np.kron(Z, I)))
H_1_3_XX = np.kron(X, np.kron(I, np.kron(X, I)))
H_1_3_YY = np.kron(Y, np.kron(I, np.kron(Y, I)))
H_1_3_ZZ = np.kron(Z, np.kron(I, np.kron(Z, I)))
H_2_3_XX = np.kron(X, np.kron(X, np.kron(I, I)))
H_2_3_YY = np.kron(Y, np.kron(Y, np.kron(I, I)))
H_2_3_ZZ = np.kron(Z, np.kron(Z, np.kron(I, I)))

def generate_hamming_weight_three(state, exclusive):
    
    Pauli_X_e0 = np.kron(X, np.kron(X, np.kron(X, I)))
    Pauli_X_e1 = np.kron(X, np.kron(X, np.kron(I, X)))
    Pauli_X_e2 = np.kron(X, np.kron(I, np.kron(X, X)))
    Pauli_X_e3 = np.kron(I, np.kron(X, np.kron(X, X)))
    
    if exclusive == 0:
        state = Pauli_X_e0 @ state
    elif exclusive == 1:
        state = Pauli_X_e1 @ state
    elif exclusive == 2:
        state = Pauli_X_e2 @ state
    elif exclusive == 3:
        state = Pauli_X_e3 @ state
        
    return state

def generate_XY_mixer(state, theta, i, j):
    if i == 0 and j == 1:
        state = expm(-1j * theta * (H_0_1_XX + H_0_1_YY)) @ state
    elif i == 0 and j == 2:
        state = expm(-1j * theta * (H_0_2_XX + H_0_2_YY)) @ state
    elif i == 0 and j == 3:
        state = expm(-1j * theta * (H_0_3_XX + H_0_3_YY)) @ state
    elif i == 1 and j == 2:
        state = expm(-1j * theta * (H_1_2_XX + H_1_2_YY)) @ state
    elif i == 1 and j == 3:
        state = expm(-1j * theta * (H_1_3_XX + H_1_3_YY)) @ state
    elif i == 2 and j == 3:
        state = expm(-1j * theta * (H_2_3_XX + H_2_3_YY)) @ state
        
    return state
        
def generate_heisenberg_glass(J_0_1, J_0_2, J_0_3, J_1_2, J_1_3, J_2_3):
    H_0_1 = J_0_1 * (H_0_1_XX + H_0_1_YY + H_0_1_ZZ)
    H_0_2 = J_0_2 * (H_0_2_XX + H_0_2_YY + H_0_2_ZZ)
    H_0_3 = J_0_3 * (H_0_3_XX + H_0_3_YY + H_0_3_ZZ)
    H_1_2 = J_1_2 * (H_1_2_XX + H_1_2_YY + H_1_2_ZZ)
    H_1_3 = J_1_3 * (H_1_3_XX + H_1_3_YY + H_1_3_ZZ)
    H_2_3 = J_2_3 * (H_2_3_XX + H_2_3_YY + H_2_3_ZZ)
    
    H_sg = H_0_1 + H_0_2 + H_0_3 + H_1_2 + H_1_3 + H_2_3
    
    return H_sg

def generate_magnetic(h_0, h_1, h_2, h_3):
    Pauli_Z_0 = h_0 * np.kron(I, np.kron(I, np.kron(I, Z)))
    Pauli_Z_1 = h_1 * np.kron(I, np.kron(I, np.kron(Z, I)))
    Pauli_Z_2 = h_2 * np.kron(I, np.kron(Z, np.kron(I, I)))
    Pauli_Z_3 = h_3 * np.kron(Z, np.kron(I, np.kron(I, I)))
    
    H_m = Pauli_Z_0 + Pauli_Z_1 + Pauli_Z_2 + Pauli_Z_3
    
    return H_m
    
def cal_distance(params, pairs):
    state = np.array(np.kron([1., 0.], np.kron([1., 0.], (np.kron([1., 0.], [1., 0.])))))
    
    state = generate_hamming_weight_three(state, 0)
    i = 0
    for pair in pairs:
        state = generate_XY_mixer(state, params[i], pair[0], pair[1])
        i += 1
    
    state_prob = []
    target_state_prob = []
    
    H_sg = generate_heisenberg_glass(-1, 2, 3, -2, -3, -4) + generate_magnetic(1, -1, 2, 3)
    eigenvalues, eigenstates = eigh(H_sg)
    target_state = eigenstates[:, np.argmin(eigenvalues)]
    
    for i in range(16):
        state_prob.append(np.abs(state[i])**2)
        target_state_prob.append(np.abs(target_state[i])**2)
        
    distance = euclidean(state_prob, target_state_prob)
    
    return distance
        
def Problem_a():
    initial_params = [np.pi / 4, np.pi / 4, np.pi / 4]
    XY_pair = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    best_value = float('inf')
    best_theta = initial_params
    best_comb = [(0, 0), (0, 0), (0, 0)]
    
    for pairs in combinations(XY_pair, 3):
        result = minimize(cal_distance, initial_params, args=(pairs,), method='Nelder-Mead')
        if best_value > result.fun:
            best_value = result.fun
            best_theta = result.x
            best_comb = pairs
        
    print("minimum euclidean distance : ", best_value)
    print("theta : ", best_theta)
    print("XY-mixer combination : ", best_comb)
    
def generate_XY_mixer_circ(circ, theta, i, j):
    circ.rxx(2*theta, i, j)
    
def Problem_b_c():
    circ = QuantumCircuit(4)
    circ.x(1)
    circ.x(2)
    circ.x(3)
    
    theta = [1.32519923, 1.2158871, 0.67324855]
    
    circ.ryy(2*theta[0], 0, 1)
    circ.rxx(2*theta[0], 0, 1)
    
    circ.ryy(2*theta[1], 1, 2)
    circ.rxx(2*theta[1], 1, 2)
    
    circ.ryy(2*theta[2], 1, 3)
    circ.rxx(2*theta[2], 1, 3)
    
    circ.rz(-np.pi / 4 , 0)
    circ.rz(np.pi / 4 , 1)
    circ.rz(-np.pi / 4 , 2)
    circ.rz(-np.pi / 4 , 3)
    
    backend = Aer.get_backend('statevector_simulator')
    result = backend.run(circ).result()
    state = result.get_statevector().data
    print("state : ", state)
    
    state_prob = []
    target_state_prob = []
    
    H_sg = generate_heisenberg_glass(-1, 2, 3, -2, -3, -4) + generate_magnetic(1, -1, 2, 3)
    eigenvalues, eigenstates = eigh(H_sg)
    target_state = eigenstates[:, np.argmin(eigenvalues)]
    
    for i in range(16):
        state_prob.append(np.abs(state[i])**2)
        target_state_prob.append(np.abs(target_state[i])**2)
        
    distance = euclidean(state_prob, target_state_prob)
    print("euclidean distance : ", distance)
    
def Problem_d(shots=4096):
    backend = Aer.get_backend('aer_simulator')
    
    circ = QuantumCircuit(4)
    circ.x(1)
    circ.x(2)
    circ.x(3)
    
    theta = [1.32519923, 1.2158871, 0.67324855]
    
    circ.ryy(2*theta[0], 0, 1)
    circ.rxx(2*theta[0], 0, 1)
    
    circ.ryy(2*theta[1], 1, 2)
    circ.rxx(2*theta[1], 1, 2)
    
    circ.ryy(2*theta[2], 1, 3)
    circ.rxx(2*theta[2], 1, 3)
    
    circ.rz(-np.pi / 4 , 0)
    circ.rz(np.pi / 4 , 1)
    circ.rz(-np.pi / 4 , 2)
    circ.rz(-np.pi / 4 , 3)
    
    #Z basis measure
    qc = circ.copy()
    
    qc.measure_all()

    counts = backend.run(qc, shots=shots).result().get_counts()
    
    ev_z = 0
    for outcome, count in counts.items():
        outcome = outcome[::-1]
        
        pairs = [((0, 1), -1), ((0, 2), 2), ((0, 3), 3), 
                 ((1, 2), -2), ((1, 3), -3), ((2, 3), -4)]
        
        for (i, j), coeff in pairs:
            bit_pair = outcome[i] + outcome[j]
            if bit_pair == '00' or bit_pair == '11':
                ev_z += coeff * count / shots
            else:
                ev_z += -coeff * count / shots
                
        pairs = [(0, 1), (1, -1), (2, 2), (3, 3)]
        
        for i, coeff in pairs:
            ev_z += -coeff * count / shots
                
    print("Z measure Energy expected value : ", ev_z)
        
    #X basis measure
    qc = circ.copy()
    
    qc.h([0, 1, 2, 3])
    
    qc.measure_all()
    
    counts = backend.run(qc, shots=shots).result().get_counts()
    
    ev_x = 0
    for outcome, count in counts.items():
        pairs = [((0, 1), -1), ((0, 2), 2), ((0, 3), 3), 
                 ((1, 2), -2), ((1, 3), -3), ((2, 3), -4)]
        
        for (i, j), coeff in pairs:
            bit_pair = outcome[i] + outcome[j]
            if bit_pair == '00' or bit_pair == '11':
                ev_x += coeff * count / shots
            else:
                ev_x += -coeff * count / shots
                
    print("X measure Energy expected value : ", ev_x)
        
    #Y basis measure
    qc = circ.copy()
    
    qc.rz(-np.pi / 2 , [0, 1, 2, 3])
    qc.h([0, 1, 2, 3])
    
    qc.measure_all()
    
    counts = backend.run(qc, shots=shots).result().get_counts()
    
    ev_y = 0
    for outcome, count in counts.items():
        pairs = [((0, 1), -1), ((0, 2), 2), ((0, 3), 3), 
                 ((1, 2), -2), ((1, 3), -3), ((2, 3), -4)]
        
        for (i, j), coeff in pairs:
            bit_pair = outcome[i] + outcome[j]
            if bit_pair == '00' or bit_pair == '11':
                ev_y += coeff * count / shots
            else:
                ev_y += -coeff * count / shots
                
    print("Y measure Energy expected value : ", ev_y)
        
    circ_ground_state_eigenvalue = ev_x + ev_y + ev_z
    
    H_sg = generate_heisenberg_glass(-1, 2, 3, -2, -3, -4) + generate_magnetic(1, -1, 2, 3)
    eigenvalues, _ = eigh(H_sg)
    ground_state_eigenvalue = np.min(eigenvalues)

    print("Total Energy expected value : ", circ_ground_state_eigenvalue)
    print("Theorical ground energy : ", ground_state_eigenvalue)
        
if __name__=='__main__':
    #Problem_a()
    #Problem_b_c()
    Problem_d()