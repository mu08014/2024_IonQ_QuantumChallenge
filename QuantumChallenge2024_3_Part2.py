import numpy as np
from scipy.linalg import eigh, expm
import matplotlib.pyplot as plt
from qiskit import *
from qiskit.visualization import plot_histogram
from qiskit_aer import Aer

def generate_mixer():
    X = np.array([[0., 1.], [1., 0.]])
    I = np.array([[1., 0.], [0., 1.]])
    
    Pauli_X_0 = np.kron(I, np.kron(I, np.kron(I, X)))
    Pauli_X_1 = np.kron(I, np.kron(I, np.kron(X, I)))
    Pauli_X_2 = np.kron(I, np.kron(X, np.kron(I, I)))
    Pauli_X_3 = np.kron(X, np.kron(I, np.kron(I, I)))
    
    H_m = Pauli_X_0 + Pauli_X_1 + Pauli_X_2 + Pauli_X_3
    
    return H_m

def generate_magnetic(h_0, h_1, h_2, h_3):
    Z = np.array([[1., 0.], [0., -1.]])
    I = np.array([[1., 0.], [0., 1.]])
    
    Pauli_Z_0 = h_0 * np.kron(I, np.kron(I, np.kron(I, Z)))
    Pauli_Z_1 = h_1 * np.kron(I, np.kron(I, np.kron(Z, I)))
    Pauli_Z_2 = h_2 * np.kron(I, np.kron(Z, np.kron(I, I)))
    Pauli_Z_3 = h_3 * np.kron(Z, np.kron(I, np.kron(I, I)))
    
    H_m = Pauli_Z_0 + Pauli_Z_1 + Pauli_Z_2 + Pauli_Z_3
    
    return H_m

def generate_heisenberg_glass(J_0_1, J_0_2, J_0_3, J_1_2, J_1_3, J_2_3):
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
        
    H_0_1 = J_0_1 * (H_0_1_XX + H_0_1_YY + H_0_1_ZZ)
    H_0_2 = J_0_2 * (H_0_2_XX + H_0_2_YY + H_0_2_ZZ)
    H_0_3 = J_0_3 * (H_0_3_XX + H_0_3_YY + H_0_3_ZZ)
    H_1_2 = J_1_2 * (H_1_2_XX + H_1_2_YY + H_1_2_ZZ)
    H_1_3 = J_1_3 * (H_1_3_XX + H_1_3_YY + H_1_3_ZZ)
    H_2_3 = J_2_3 * (H_2_3_XX + H_2_3_YY + H_2_3_ZZ)
    
    H_sg = H_0_1 + H_0_2 + H_0_3 + H_1_2 + H_1_3 + H_2_3
    
    return H_sg

def generate_hamiltonian(H_sg, H_m, s):
    H = (1 - s)*H_m + s*H_sg
    return H

def generate_start_state(circ):
    circ_num = 0
    for i in range(4):
        circ.h(i)
        circ_num += 1
    return circ_num

def generate_mixer_circ(circ, s, ds):
    circ_num = 0
    for i in range(4):
        circ.rx(2*(1 - s)*ds, i)
        circ_num += 1
    return circ_num
        
        
def generate_magnetic_circ(circ, s, ds):
    circ.rz(s*2*ds, 0)
    circ.rz(-1*s*2*ds, 1)
    circ.rz(2*s*2*ds, 2)
    circ.rz(3*s*2*ds, 3)
    return 4
        
def generate_heisenberg_glass_circ(circ, s, ds):
    circ.rxx(-1*s*2*ds, 0, 1)
    circ.ryy(-1*s*2*ds, 0, 1)
    circ.rzz(-1*s*2*ds, 0, 1)
    
    circ.rxx(2*s*2*ds, 0, 2)
    circ.ryy(2*s*2*ds, 0, 2)
    circ.rzz(2*s*2*ds, 0, 2)
    
    circ.rxx(3*s*2*ds, 0, 3)
    circ.ryy(3*s*2*ds, 0, 3)
    circ.rzz(3*s*2*ds, 0, 3)
    
    circ.rxx(-2*s*2*ds, 1, 2)
    circ.ryy(-2*s*2*ds, 1, 2)
    circ.rzz(-2*s*2*ds, 1, 2)
    
    circ.rxx(-3*s*2*ds, 1, 3)
    circ.ryy(-3*s*2*ds, 1, 3)
    circ.rzz(-3*s*2*ds, 1, 3)
    
    circ.rxx(-4*s*2*ds, 2, 3)
    circ.ryy(-4*s*2*ds, 2, 3)
    circ.rzz(-4*s*2*ds, 2, 3)
    
    return 18
    
def Problem_a():
    s_values = []
    all_eigenvalues = []
    H_m = generate_mixer()
    
    for i in range(1000):
        s = 0 if i == 0 else i / 1000
        spectrums = []
        
        J = []
        J_class1 = [-1, -1, -1, -1, -1, -1]
        J_class2 = [1, -1, -1, -1, -1, -1]
        J_class3 = [1, 1, -1, -1, -1, -1]
        J_class4 = [1, 1, 1, -1, -1, -1]
        J_class5 = [-1, -1, 1, 1, -1, -1]
        J_class6 = [1, -1, 1, 1, -1, -1]
        
        J.append(J_class1)
        J.append(J_class2)
        J.append(J_class3)
        J.append(J_class4)
        J.append(J_class5)
        J.append(J_class6)
        
        for num in range(6):
            J_0_1 = J[num][0]
            J_0_2 = J[num][1]
            J_0_3 = J[num][2]
            J_1_2 = J[num][3]
            J_1_3 = J[num][4]
            J_2_3 = J[num][5]
            
            H_sg = generate_heisenberg_glass(J_0_1, J_0_2, J_0_3, J_1_2, J_1_3, J_2_3)
            
            H = (1 - s)*H_m + s*H_sg
            
            eigenvalues = np.sort(eigh(H)[0])
            spectrums.append(eigenvalues)
        
        s_values.append(s)
        all_eigenvalues.append(spectrums)
        
    plt.figure(figsize=(12, 8))

    for j in range(6):
        for i in range(len(all_eigenvalues[0][j])):
            y_values = [eigenvalues[j][i] for eigenvalues in all_eigenvalues]
            plt.plot(s_values, y_values)

        plt.xlabel('s')
        plt.ylabel('Eigenspectrum')
        plt.title(f'Part 2-a) Class {j+1} spectrums')
        plt.show()
        
def Problem_b():
    s_values = []
    all_eigenvalues = []
    
    H_sg = generate_heisenberg_glass(-1, 2, 3, -2, -3, -4) + generate_magnetic(1, -1, 2, 3)
    H_m = generate_mixer()
    
    for i in range(1000):
        s = 0 if i == 0 else i / 1000
        
        H = generate_hamiltonian(H_sg, H_m, s)

        eigenvalues = np.sort(eigh(H)[0])
        s_values.append(s)
        all_eigenvalues.append(eigenvalues)
        
    plt.figure(figsize=(12, 8))

    for i in range(len(all_eigenvalues[0])):
        y_values = [eigenvalues[i] for eigenvalues in all_eigenvalues]
        plt.plot(s_values, y_values)

    plt.xlabel('s')
    plt.ylabel('Eigenvalues')
    plt.title('Part 2-b) spcetrums')
    plt.tight_layout()
    plt.show()
    
def Problem_c_d(t, N):
    ds = t / N
    s = 0
    
    H_sg = generate_heisenberg_glass(-1, 2, 3, -2, -3, -4) + generate_magnetic(1, -1, 2, 3)
    H_m = generate_mixer()
    
    eigenvalues, eigenstates = np.linalg.eigh(H_m)
    state = eigenstates[:, np.argmin(eigenvalues)]
    
    while (s < t):
        H = generate_hamiltonian(H_sg, H_m, s / t)
        U = expm(-1j * H * ds)
        
        state = U @ state
        s += ds
        
    eigenvalues, eigenstates = np.linalg.eigh(H_sg)
    target_state = eigenstates[:, np.argmin(eigenvalues)]
    
    #similarity
    #print(np.abs(np.vdot(state, target_state))**2)
    
    state_prob = []
    target_state_prob = []
    
    for i in range(16):
        state_prob.append(np.abs(state[i])**2)
        target_state_prob.append(np.abs(target_state[i])**2)
        
    if (np.allclose(state_prob, target_state_prob, atol=1e-3)):
        print('success')
    else:
        print('failed')

def Problem_e_f(t, N, shots=4096):
    ds = t / N
    s = 0
    single_qubit_gate_num = 0
    two_qubits_gate_num = 0
    
    circ = QuantumCircuit(4)
    single_qubit_gate_num += generate_start_state(circ)
    
    while (s < t):
        two_qubits_gate_num += generate_heisenberg_glass_circ(circ, s / t, ds)
        single_qubit_gate_num += generate_magnetic_circ(circ, s / t, ds)
        single_qubit_gate_num += generate_mixer_circ(circ, s / t, ds)
        s += ds    
    
    circ.measure_all()
    backend = Aer.get_backend(name='aer_simulator')
    counts = backend.run(circ, shots=shots).result().get_counts()
    
    H_sg = generate_heisenberg_glass(-1, 2, 3, -2, -3, -4) + generate_magnetic(1, -1, 2, 3)
    eigenvalues, eigenstates = np.linalg.eigh(H_sg)
    target_state = eigenstates[:, np.argmin(eigenvalues)]
    
    state_prob = []
    target_state_prob = []
    
    for i in range(16):
        binary_string = bin(i)[2:].zfill(4)
        if counts.get(binary_string, 0) != 0:
            state_prob.append(counts[binary_string] / shots)
        else:
            state_prob.append(0)
        target_state_prob.append(np.abs(target_state[i])**2)
    
    #similarity
    #print(np.abs(np.vdot(state_prob, target_state_prob)))
    
    print("all number of single qubit gates : ", single_qubit_gate_num)
    print("all number of two qubits gates : ", two_qubits_gate_num)
    
    if (np.allclose(state_prob, target_state_prob, atol=1e-3)):
        print('success')
    else:
        print('failed')
    
if __name__=='__main__':
    #Problem_a()
    #Problem_b()
    #Problem_c_d(900, 900)
    Problem_e_f(900, 900)