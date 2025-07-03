import numpy as np
from qiskit import *
from qiskit.visualization import plot_histogram
from qiskit_aer import Aer
import matplotlib.pyplot as plt

def give_score(U_hidden, U_gen) :
    backend2 = Aer.get_backend('statevector_simulator')
    qc = U_hidden.compose(U_gen.inverse())
    result = backend2.run(qc).result()
    score = result.get_statevector()[0]
    return np.abs(score)

def generate_clifford_circ(num_qubits, depth):
    U_hidden = QuantumCircuit(num_qubits)
    
    for i in range(num_qubits):
        U_hidden.h(i)
        
    count = 0
    
    s1s2 = {-1, -1}
    while (count < depth):
        site_i = int(np.random.rand() * num_qubits)
        site_j = int(np.random.rand() * num_qubits)
        if site_i == site_j or {site_i, site_j} == s1s2:
            continue
        s1s2 = {site_i, site_j}
        
        count += 1
        U_hidden.cz(site_i, site_j)
        
    return U_hidden

def generate_Ugen(U_hidden, num_qubits, shots=100, threshold=0):
    backend = Aer.get_backend(name='aer_simulator')
    score = np.zeros((num_qubits, num_qubits), dtype=int)
    
    edge = np.zeros((num_qubits, num_qubits))
    for j in range(num_qubits):
        edge[j][j] = 1
    
    for i in range(num_qubits):
        qc1 = QuantumCircuit(num_qubits)
        qc1.compose(U_hidden, inplace=True)
        qc1.h(i)
        qc1.measure_all()
         
        counts = backend.run(qc1, shots=shots).result().get_counts()
        counts = dict(sorted(counts.items(), key=lambda item: item[0].count('1')))
        
        for bit_string, value in counts.items():
            if value > threshold:
                one_bit_count = 0
                valid_one_bit_count = 0
                
                #count 1 bit
                for j in range(num_qubits):
                    if bit_string[j] == '1':
                        one_bit_count += 1
                        if edge[i][num_qubits-1-j] == 1:
                            valid_one_bit_count += 1
                
                if one_bit_count == 1:
                    for j in range(num_qubits):
                        if bit_string[j] == '1' and j != num_qubits-1-i:
                            edge[i][num_qubits-1-j] = -1
                            break
                elif one_bit_count - valid_one_bit_count == 1:
                    for j in range(num_qubits):
                        if bit_string[j] == '1' and edge[i][num_qubits-1-j] == 0:
                            edge[i][num_qubits-1-j] = 1
                            break
                        
        for j in range(num_qubits):
            if edge[i][j] == 0:
                edge[i][j] = -1
                    
    cz_qubits = []
                    
    for i in range(0, num_qubits-1):
        for j in range(i+1, num_qubits):
            if edge[i][j] == 1:
                cz_qubits.append((i, j))
                
    
    U_gen = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        U_gen.h(i)
    
    for i, j in cz_qubits:
        U_gen.cz(i, j)
    
    return U_gen

if __name__ == '__main__':
    num_qubits = 5
    depth = 5
    U_hidden = generate_clifford_circ(num_qubits, depth)
    U_hidden.draw('mpl')
    plt.show()
    U_gen = generate_Ugen(U_hidden, num_qubits)
    U_gen.draw('mpl')
    plt.show()
    print(give_score(U_hidden, U_gen))
    
    