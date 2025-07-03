import numpy as np
from qiskit import *
from qiskit.visualization import plot_histogram
from qiskit_aer import Aer
import matplotlib.pyplot as plt

def create_2q_target_circuit(qc_data):
    qc = QuantumCircuit( 2)
    for i in range ( 2):
        qc.ry(qc_data[ 2*i+ 1],i)
        qc.rz(qc_data[ 2*i+ 1],i)
    qc.cx( 0, 1)
    qc.rz(qc_data[ 4], 1)
    qc.cx( 0, 1)
    for i in range ( 2):
        qc.ry(qc_data[4+2*i + 1],i)
        qc.rz(qc_data[4+2*i + 1],i)
    return qc

def give_score(U_hidden, U_gen) :
    backend2 = Aer.get_backend('statevector_simulator')
    qc = U_hidden.compose(U_gen.inverse())
    result = backend2.run(qc).result()
    score = result.get_statevector()[ 0]
    return np.abs(score)

def draw_histogram(circuit):
    backend = Aer.get_backend(name='aer_simulator')
    counts = backend.run(circuit, shots=100).result().get_counts()
    plot_histogram(counts)
    plt.show()
    
def probability_gen(counts, shots):
    if counts.get('00', 0) == 0:
        p00 = 0
    else:
        p00 = counts.get('00', 0) / shots
    if counts.get('10', 0) == 0:
        p01 = 0
    else:
        p01 = counts.get('10', 0) / shots
    if counts.get('01', 0) == 0:
        p10 = 0
    else:
        p10 = counts.get('01', 0) / shots
    if counts.get('11', 0) == 0:
        p11 = 0
    else:
        p11 = counts.get('11', 0) / shots
    return p00, p01, p10, p11
    
def make_two_qubit_Ugen(U_hidden, shots=100):
    backend = Aer.get_backend(name='aer_simulator')
    
    #measure ZZ
    qc1 = U_hidden.copy()
    qc1.measure_all()
    counts = backend.run(qc1, shots=shots).result().get_counts()
        
    if counts.get('00', 0) == 0:
        a = 0
    else:
        a = np.sqrt(counts.get('00', 0) / shots)
    if counts.get('10', 0) == 0:
        b = 0
    else:
        b = np.sqrt(counts.get('10', 0) / shots)
    if counts.get('01', 0) == 0:
        c = 0
    else:
        c = np.sqrt(counts.get('01', 0) / shots)
    if counts.get('11', 0) == 0:
        d = 0
    else:
        d = np.sqrt(counts.get('11', 0) / shots)
    
    #measure HIZZ
    qc1 = U_hidden.copy()
    qc2 = QuantumCircuit(2)
    qc2.h(0)
    circ = qc1.compose(qc2)
    circ.measure_all()
    counts = backend.run(circ, shots=shots).result().get_counts()
    
    p00, p01, p10, p11 = probability_gen(counts, shots)

    cos_theta2 = (p00 - p10) / (2*a*c)
    
    cos_theta13_plus_sin_theta13 = (p01 - p11) / (2*b*d)
    
    #measure IHZZ
    qc1 = U_hidden.copy()
    qc2 = QuantumCircuit(2)
    qc2.h(1)
    circ = qc1.compose(qc2)
    circ.measure_all()
    counts = backend.run(circ, shots=shots).result().get_counts()
    
    p00, p01, p10, p11 = probability_gen(counts, shots)
    
    cos_theta1 = (p00 - p01) / (2*a*b)
    
    cos_theta23_plus_sin_theta23 = (p10 - p11) / (2*c*d)
    
    #measure HISIZZ
    qc1 = U_hidden.copy()
    qc2 = QuantumCircuit(2)
    qc2.s(0)
    qc2.h(0)
    circ = qc1.compose(qc2)
    circ.measure_all()
    counts = backend.run(circ, shots=shots).result().get_counts()
    
    p00, p01, p10, p11 = probability_gen(counts, shots)
    
    sin_theta2 = (p00 - p10) / (-2*a*c)
    
    sin_theta1_cos_theta3_minus_sin_theta3_cos_theta1 = (p01 - p11) / (2*b*d)
    
    #measure IHISZZ
    qc1 = U_hidden.copy()
    qc2 = QuantumCircuit(2)
    qc2.s(1)
    qc2.h(1)
    circ = qc1.compose(qc2)
    circ.measure_all()
    counts = backend.run(circ, shots=shots).result().get_counts()
    
    p00, p01, p10, p11 = probability_gen(counts, shots)
    
    sin_theta1 = (p00 - p01) / (-2*a*b)
    
    sin_theta2_cos_theta3_minus_sin_theta3_cos_theta2 = (p10 - p11) / (2*c*d)
    
    #calculate theta
    theta1 = np.arctan2(sin_theta1, cos_theta1)
    theta2 = np.arctan2(sin_theta2, cos_theta2)
        
    sin_theta3_0 = cos_theta13_plus_sin_theta13*np.sin(theta1) - sin_theta1_cos_theta3_minus_sin_theta3_cos_theta1*np.cos(theta1)
    sin_theta3_1 = cos_theta23_plus_sin_theta23*np.sin(theta2) - sin_theta2_cos_theta3_minus_sin_theta3_cos_theta2*np.cos(theta2)
    sin_theta3 = (sin_theta3_0 + sin_theta3_1) / 2
    
    cos_theta3_0 = cos_theta13_plus_sin_theta13*np.cos(theta1) + sin_theta1_cos_theta3_minus_sin_theta3_cos_theta1*np.sin(theta1)
    cos_theta3_1 = cos_theta23_plus_sin_theta23*np.cos(theta2) + sin_theta2_cos_theta3_minus_sin_theta3_cos_theta2*np.sin(theta2)
    cos_theta3 = (cos_theta3_0 + cos_theta3_1) / 2
    
    theta3 = np.arctan2(sin_theta3, cos_theta3)
    
    if theta1 < 0:
        theta1 += 2 * np.pi
    if theta2 < 0:
        theta2 += 2 * np.pi
    if theta3 < 0:
        theta3 += 2 * np.pi
    
    #measure HHZZ
    qc1 = U_hidden.copy()
    qc2 = QuantumCircuit(2)
    qc2.h(0)
    qc2.h(1)
    circ = qc1.compose(qc2)
    circ.measure_all()
    counts = backend.run(circ, shots=shots).result().get_counts()
    
    p00, p01, p10, p11 = probability_gen(counts, shots)
    
    cos_theta1_HHZZ = (2*(p00 + p10) - 1 - 2*c*d*np.cos(theta2 - theta3)) / (2*a*b)
    cos_theta2_HHZZ = (2*(p00 + p01) - 1 - 2*b*d*np.cos(theta1 - theta3)) / (2*a*c)
    cos_theta3_HHZZ = (2*(p00 + p11) - 1 - 2*b*c*np.cos(theta1 - theta2)) / (2*a*d)
    
    #measure HHSIZZ
    
    qc1 = U_hidden.copy()
    qc2 = QuantumCircuit(2)
    qc2.s(0)
    qc2.h(0)
    qc2.h(1)
    circ = qc1.compose(qc2)
    circ.measure_all()
    counts = backend.run(circ, shots=shots).result().get_counts()
    
    p00, p01, p10, p11 = probability_gen(counts, shots)
    
    cos_theta1_HHSIZZ = (2*(p00 + p10) - 1 - 2*c*d*np.cos(theta2 - theta3)) / (2*a*b)
    sin_theta2_HHSIZZ = (2*(p00 + p01) - 1 - 2*b*d*np.sin(theta1 - theta3)) / (-2*a*c)
    sin_theta3_HHSIZZ = (2*(p00 + p11) - 1 - 2*b*c*np.sin(theta1 - theta2)) / (-2*a*d)
    
    #update theta
    theta2_cand = np.arctan2(sin_theta2_HHSIZZ, cos_theta2_HHZZ)
    if theta2_cand < 0:
        theta2_cand += 2*np.pi
    theta2 = (theta2 + theta2_cand)/2
    
    theta3_cand = np.arctan2(sin_theta3_HHSIZZ, cos_theta3_HHZZ)
    if theta3_cand < 0:
        theta3_cand += 2*np.pi
    theta3 = (theta3 + theta3_cand)/2
    
    #measure HHISZZ
    
    qc1 = U_hidden.copy()
    qc2 = QuantumCircuit(2)
    qc2.s(1)
    qc2.h(0)
    qc2.h(1)
    circ = qc1.compose(qc2)
    circ.measure_all()
    counts = backend.run(circ, shots=shots).result().get_counts()
    
    p00, p01, p10, p11 = probability_gen(counts, shots)
    
    sin_theta1_HHISZZ = (2*(p00 + p10) - 1 - 2*c*d*np.sin(theta2 - theta3)) / (-2*a*b)
    cos_theta2_HHISZZ = (2*(p00 + p01) - 1 - 2*b*d*np.cos(theta1 - theta3)) / (2*a*c)
    sin_theta3_HHISZZ = (2*(p00 + p11) - 1 - 2*b*c*np.sin(theta1 - theta2)) / (-2*a*d)
    
    #update theta
    theta1_cand = np.arctan2(sin_theta1_HHISZZ, cos_theta1_HHZZ)
    if theta1_cand < 0:
        theta1_cand += 2*np.pi
    theta1 = (theta1 + theta1_cand)/2
    
    #measure HHSSZZ
    
    qc1 = U_hidden.copy()
    qc2 = QuantumCircuit(2)
    qc2.s(0)
    qc2.s(1)
    qc2.h(0)
    qc2.h(1)
    circ = qc1.compose(qc2)
    circ.measure_all()
    counts = backend.run(circ, shots=shots).result().get_counts()
    
    p00, p01, p10, p11 = probability_gen(counts, shots)
    
    sin_theta1_HHSSZZ = (2*(p00 + p10) - 1 - 2*c*d*np.sin(theta2 - theta3)) / (-2*a*b)
    sin_theta2_HHSSZZ = (2*(p00 + p01) - 1 - 2*b*d*np.sin(theta1 - theta3)) / (-2*a*c)
    cos_theta3_HHSSZZ = (2*(p00 + p11) - 1 - 2*b*c*np.cos(theta1 - theta2)) / (-2*a*d)
    
    #update theta
    theta1_cand = np.arctan2(sin_theta1_HHSSZZ, cos_theta1_HHSIZZ)
    if theta1_cand < 0:
        theta1_cand += 2*np.pi
    theta1 = (theta1 + theta1_cand)/2
    
    theta2_cand = np.arctan2(sin_theta2_HHSSZZ, cos_theta2_HHISZZ)
    if theta2_cand < 0:
        theta2_cand += 2*np.pi
    theta2 = (theta2 + theta2_cand)/2
    
    theta3_cand = np.arctan2(sin_theta3_HHISZZ, cos_theta3_HHSSZZ)
    if theta3_cand < 0:
        theta3_cand += 2*np.pi
    theta3 = (theta3 + theta3_cand)/2
    
    #generate Ugen
    Ugen = QuantumCircuit(2)
    
    theta_a = 2 * np.arccos(a)
    theta_b = 2 * np.arcsin(b / np.sqrt(b**2 + c**2 + d**2))
    
    Ugen.ry(theta_a, 0)

    Ugen.ry(theta_b, 1)

    Ugen.rz(theta1, 1)
    Ugen.rz(theta2, 0)
    Ugen.cx(0, 1)
    Ugen.rz(theta3, 1)
    
    return Ugen
    
if __name__ == '__main__':
    for _ in range(10):
        random_vec = np.random.rand(9)* 2.0 * np.pi
            
        U_hidden = create_2q_target_circuit(random_vec)
        U_gen = make_two_qubit_Ugen(U_hidden)
        backend = Aer.get_backend(name='aer_simulator')
        
        #measure ZZ
        print(give_score(U_hidden, U_gen))
    
    
    
    