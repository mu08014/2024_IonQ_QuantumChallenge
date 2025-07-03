import numpy as np
from qiskit import *
from qiskit.visualization import plot_histogram
from qiskit_aer import Aer
import matplotlib.pyplot as plt

def create_1q_target_circuit(qc_data):
    qc = QuantumCircuit(1)
    qc.ry(qc_data[ 0], 0)
    qc.rz(qc_data[ 1], 0)
    return qc

def give_score(U_hidden, U_gen) :
    backend2 = Aer.get_backend('statevector_simulator')
    qc = U_hidden.compose(U_gen.inverse())
    result = backend2.run(qc).result()
    score = result.get_statevector()[ 0]
    return np.abs(score)

def draw_histogram(circuit):
    backend = Aer.get_backend(name='aer_simulator')
    counts = backend.run(circuit, shots=1000).result().get_counts()
    plot_histogram(counts)
    plt.show()
    
def make_one_qubit_Ugen(U_hidden, shots=100):
    backend = Aer.get_backend(name='aer_simulator')
    
    qc1 = U_hidden.copy()
    qc1.measure_all()
    counts = backend.run(qc1, shots=shots).result().get_counts()
    
    if counts.get('0', 0) == 0:
        a1 = 0
    elif counts.get('1', 0) == 0:
        a1 = 1
    else:
        a1 = np.sqrt(counts.get('0', 0) / shots)
    
    qc1 = U_hidden.copy()
    qc2 = QuantumCircuit(1)
    qc2.s(0)
    circ = qc1.compose(qc2)
    circ.measure_all()
    counts = backend.run(circ, shots=shots).result().get_counts()
    
    if counts.get('0', 0) == 0:
        a2 = 0
    elif counts.get('1', 0) == 0:
        a2 = 1
    else:
        a2 = np.sqrt(counts.get('0', 0) / shots)
        
    a = (a1 + a2) / 2
    b = np.sqrt(1 - a**2)
    
    qc1 = U_hidden.copy()
    qc2 = QuantumCircuit(1)
    qc2.h(0)
    circ = qc1.compose(qc2)
    circ.measure_all()
    counts = backend.run(circ, shots=shots).result().get_counts()
    
    if counts.get('0', 0) == 0:
        prob = 0
    else:
        prob = counts.get('0', 0) / shots
    
    if a == 0 or b == 0:
        cos_theta = 1
    else:    
        cos_theta = (2 * prob - 1) / (2 * a * b)
    
    qc1 = U_hidden.copy()
    qc2 = QuantumCircuit(1)
    qc2.s(0)
    qc2.h(0)
    circ = qc1.compose(qc2)
    circ.measure_all()
    counts = backend.run(circ, shots=shots).result().get_counts()
    
    if counts.get('0', 0) == 0:
        prob = 0
    else:
        prob = counts.get('0', 0) / shots
    
    if a == 0 or b == 0:
        sin_theta = 0
    else:
        sin_theta = (2 * prob - 1) / (-2 * a * b)
        
    t1 = np.arctan2(sin_theta, cos_theta)
    
    if t1 < 0:
        t1 += 2 * np.pi
    
    qc1 = U_hidden.copy()
    qc2 = QuantumCircuit(1)
    qc2.s(0)
    qc2.s(0)
    qc2.h(0)
    circ = qc1.compose(qc2)
    circ.measure_all()
    counts = backend.run(circ, shots=shots).result().get_counts()
    if counts.get('0', 0) == 0:
        prob = 0
    else:
        prob = counts.get('0', 0) / shots
    
    if a == 0 or b == 0:
        m_cos_theta = 0
    else:
        m_cos_theta = (2 * prob - 1) / (-2 * a * b)
        
    qc1 = U_hidden.copy()
    qc2 = QuantumCircuit(1)
    qc2.s(0)
    qc2.s(0)
    qc2.s(0)
    qc2.h(0)
    circ = qc1.compose(qc2)
    circ.measure_all()
    counts = backend.run(circ, shots=shots).result().get_counts()
    if counts.get('0', 0) == 0:
        prob = 0
    else:
        prob = counts.get('0', 0) / shots
    
    if a == 0 or b == 0:
        m_sin_theta = 0
    else:
        m_sin_theta = (2 * prob - 1) / (2 * a * b)
        
    t2 = np.arctan2(m_sin_theta, m_cos_theta)
    
    if t2 < 0:
        t2 += 2 * np.pi
        
    theta = (t1 + t2) / 2
    if theta < 0:
        theta += 2 * np.pi
    
    Ugen = QuantumCircuit(1)
    Ugen.ry(2 * np.arccos(a), 0)
    Ugen.rz(theta, 0)
    
    return Ugen
    
if __name__ == '__main__':
    random_vec = np.random.rand(2 )* 2.0 * np.pi
        
    U_hidden = create_1q_target_circuit(random_vec)
    U_gen = make_one_qubit_Ugen(U_hidden)
    print(give_score(U_hidden, U_gen))
    
    
    
    