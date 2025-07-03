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

def low_depth(num_qubits):
    qc = QuantumCircuit(num_qubits)
    for i in range(0, num_qubits-1, 2):
        params = np.random.rand(15)*2*np.pi
        qc.u(params[0], params[1], params[2], i)
        qc.u(params[3], params[4], params[5], i+1)
        qc.rxx(params[6],i,i+1)
        qc.ryy(params[7],i,i+1)
        qc.rzz(params[8],i,i+1)
        qc.u(params[9], params[10], params[11], i)
        qc.u(params[12], params[13], params[14], i+1)
        
    for i in range(1, num_qubits-(num_qubits+1)%2, 2):
        params = np.random.rand(15)*2*np.pi
        qc.u(params[0], params[1], params[2], i)
        qc.u(params[3], params[4], params[5], i+1)
        qc.rxx(params[6],i,i+1)
        qc.ryy(params[7],i,i+1)
        qc.rzz(params[8],i,i+1)
        qc.u(params[9], params[10], params[11], i)
        qc.u(params[12], params[13], params[14], i+1)
        
    U_hidden = qc
    return U_hidden

def generate_Ugen(U_hidden, num_qubits, shots=100):
    backend = Aer.get_backend(name='aer_simulator')
    score = np.zeros((num_qubits, 2), dtype=int)
    Ugen = QuantumCircuit(num_qubits)
    
    #Z measure
    qc1 = QuantumCircuit(num_qubits)
    qc1.compose(U_hidden, inplace=True)
    qc1.measure_all()
         
    counts = backend.run(qc1, shots=shots).result().get_counts()
    
    a = []
    b = []
        
    for bit_string, value in counts.items():
        for i in range(num_qubits):
            if bit_string[i] == '0':
                score[num_qubits-1-i][0] += value
            else:
                score[num_qubits-1-i][1] += value
    
    for i in range(num_qubits):
        if score[i][1] == 0:
            prob = 0
        else:
            prob = score[i][1]/shots
        
        theta = 2 * np.arcsin(np.sqrt(prob))
        if theta < 0:
            theta += 2*np.pi
        a.append(np.cos(theta/2))
        b.append(np.sin(theta/2))
        Ugen.ry(theta, i)

    #X measure
    qc1 = U_hidden.copy()
    qc2 = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc2.h(i)
    circ = qc1.compose(qc2)
    circ.measure_all()
         
    counts = backend.run(circ, shots=shots).result().get_counts()
    cos_theta = []
    
    for i in range(num_qubits):
        for j in range(2):
            score[i][j] = 0
        
    for bit_string, value in counts.items():
        for i in range(num_qubits):
            if bit_string[i] == '0':
                score[num_qubits-1-i][0] += value
            else:
                score[num_qubits-1-i][1] += value
                
    for i in range(num_qubits):
        if score[i][0] == 0:
            prob = 0
        else:
            prob = score[i][0]/shots
        
        if a[i] == 0 or b[i] == 0:
            cos_theta.append(1)
        else:
            cos_theta.append((2*prob - 1) / (2*a[i]*b[i]))
    
    #Y measure
    qc1 = U_hidden.copy()
    qc2 = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc2.s(i)
        qc2.h(i)
    circ = qc1.compose(qc2)
    circ.measure_all()
         
    counts = backend.run(circ, shots=shots).result().get_counts()
    sin_theta = []
    
    for i in range(num_qubits):
        for j in range(2):
            score[i][j] = 0
        
    for bit_string, value in counts.items():
        for i in range(num_qubits):
            if bit_string[i] == '0':
                score[num_qubits-1-i][0] += value
            else:
                score[num_qubits-1-i][1] += value
                
    for i in range(num_qubits):
        if score[i][0] == 0:
            prob = 0
        else:
            prob = score[i][0]/shots
        
        if a[i] == 0 or b[i] == 0:
            sin_theta.append(0)
        else:
            sin_theta.append((2 * prob - 1) / (-2 * a[i] * b[i]))
            
    #SSHZ measure
    qc1 = U_hidden.copy()
    qc2 = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc2.s(i)
        qc2.s(i)
        qc2.h(i)
    circ = qc1.compose(qc2)
    circ.measure_all()
         
    counts = backend.run(circ, shots=shots).result().get_counts()
    m_cos_theta = []
    
    for i in range(num_qubits):
        for j in range(2):
            score[i][j] = 0
        
    for bit_string, value in counts.items():
        for i in range(num_qubits):
            if bit_string[i] == '0':
                score[num_qubits-1-i][0] += value
            else:
                score[num_qubits-1-i][1] += value
                
    for i in range(num_qubits):
        if score[i][0] == 0:
            prob = 0
        else:
            prob = score[i][0]/shots
        
        if a[i] == 0 or b[i] == 0:
            m_cos_theta.append(1)
        else:
            m_cos_theta.append((2*prob - 1) / (-2*a[i]*b[i]))
            
    #SSSHZ measure
    qc1 = U_hidden.copy()
    qc2 = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc2.s(i)
        qc2.s(i)
        qc2.s(i)
        qc2.h(i)
    circ = qc1.compose(qc2)
    circ.measure_all()
         
    counts = backend.run(circ, shots=shots).result().get_counts()
    m_sin_theta = []
    
    for i in range(num_qubits):
        for j in range(2):
            score[i][j] = 0
        
    for bit_string, value in counts.items():
        for i in range(num_qubits):
            if bit_string[i] == '0':
                score[num_qubits-1-i][0] += value
            else:
                score[num_qubits-1-i][1] += value
                
    for i in range(num_qubits):
        if score[i][0] == 0:
            prob = 0
        else:
            prob = score[i][0]/shots
        
        if a[i] == 0 or b[i] == 0:
            m_sin_theta.append(1)
        else:
            m_sin_theta.append((2*prob - 1) / (2*a[i]*b[i]))
    
    #calculate theta        
    for i in range(num_qubits):
        theta = np.arctan2(sin_theta[i], cos_theta[i])
        m_theta = np.arctan2(m_sin_theta[i], m_cos_theta[i])
        if theta < 0:
            theta += 2*np.pi
        if m_theta < 0:
            m_theta += 2*np.pi
        theta = (theta + m_theta)/2
        Ugen.rz(theta, i)
    
    return Ugen

if __name__ == '__main__':
    #for i in range(100):
        num_qubits = 5
        U_hidden = low_depth(num_qubits)
        U_gen = generate_Ugen(U_hidden, num_qubits)
        print(give_score(U_hidden, U_gen))