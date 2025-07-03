import numpy as np
from qiskit import *
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from math import log2
from qiskit_aer import Aer

def hamming_weight_circuit(state_vector, num_qubits):
    k = int(np.ceil(np.log2(num_qubits + 1)))
    w = pow(2, k) - (num_qubits + 1)
    N = pow(2, k) - 1
    
    num_classical_bits = k
    
    qc = QuantumCircuit(k + num_qubits + w, num_classical_bits)
    
    #initialize by state vector
    qc.h(range(k))
    qc.initialize(state_vector, range(k, k + num_qubits))
    
    #Rz
    for j in range(k):
        qc.rz((np.pi*N*pow(2, j)) / (N+1), j)
    
    #control-Rz
    for j in range(k):
        for target in range(num_qubits + w):
            qc.crz((2*np.pi*pow(2, k-1-j)) / (N+1), k-1-j, k + target)
    
    #Inverse QFT
    for j in range(k // 2):
        qc.swap(j, k-j-1)
    for j in range(k):
        for m in range(j):
            qc.cp(-np.pi/float(2**(j-m)), m, j)
        qc.h(j)
    
    #measure
    qc.measure(range(k), range(k))
        
    return qc

def calculate_hamming_weight(result, shots):
    hamming_weight = 0
    
    for bit_string, value in result.items():
        hw = int(bit_string, 2)
        hamming_weight += hw * value / shots
    return hamming_weight

if __name__ == '__main__':
    shots = 1024
    a_state_vector = [0, 0, 0, 1]
    b_state_vector = [0, 1, 0, 0]
    c_state_vector = [1/np.sqrt(2), 0, 0, 1/np.sqrt(2)]
    d_state_vector = [0, 0, 0, 1/np.sqrt(3), 0, 1/np.sqrt(3), 1/np.sqrt(3), 0]
    
    #3-a)
    num_qubits = int(log2(len(a_state_vector)))
    qc = hamming_weight_circuit(a_state_vector, num_qubits)
    
    backend = Aer.get_backend('aer_simulator')
    tqc = transpile(qc, backend)
    results = backend.run(tqc, shots=shots).result().get_counts()
    
    print("a) Hamming weight : ", calculate_hamming_weight(results, shots))
    
    #3-b)
    num_qubits = int(log2(len(b_state_vector)))
    qc = hamming_weight_circuit(b_state_vector, num_qubits)
    
    backend = Aer.get_backend('aer_simulator')
    tqc = transpile(qc, backend)
    results = backend.run(tqc, shots=shots).result().get_counts()
    
    print("b) Hamming weight : ", calculate_hamming_weight(results, shots))
    
    #3-c)
    num_qubits = int(log2(len(c_state_vector)))
    qc = hamming_weight_circuit(c_state_vector, num_qubits)
    
    backend = Aer.get_backend('aer_simulator')
    tqc = transpile(qc, backend)
    results = backend.run(tqc, shots=shots).result().get_counts()
    
    print("c) Hamming weight : ", calculate_hamming_weight(results, shots))
    
    #3-d)
    num_qubits = int(log2(len(d_state_vector)))
    qc = hamming_weight_circuit(d_state_vector, num_qubits)
    
    backend = Aer.get_backend('aer_simulator')
    tqc = transpile(qc, backend)
    results = backend.run(tqc, shots=shots).result().get_counts()
    
    print("d) Hamming weight : ", calculate_hamming_weight(results, shots))
    
    
    
    
    