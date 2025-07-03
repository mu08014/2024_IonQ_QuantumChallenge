import numpy as np
from scipy.linalg import eigh

def generate_hamiltonian_eigenvalue():
    #generate pauli matrix
    X = np.array([[0., 1.], [1., 0.]])
    Y = np.array([[0., -1j], [1j, 0.]])
    Z = np.array([[1., 0.], [0., -1.]])
    I = np.array([[1., 0.], [0., 1.]])
    
    #generate spin interaction
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
    
    spectrums = []
    m_spectrums = []
    
    #gen hamiltonian and cal eigenvalue
    for num in range(64):
        binary_string = bin(num)[2:].zfill(6)
        
        J_0_1 = 2 * int(binary_string[5]) - 1
        J_0_2 = 2 * int(binary_string[4]) - 1
        J_0_3 = 2 * int(binary_string[3]) - 1
        J_1_2 = 2 * int(binary_string[2]) - 1
        J_1_3 = 2 * int(binary_string[1]) - 1
        J_2_3 = 2 * int(binary_string[0]) - 1
        
        H_0_1 = J_0_1 * (H_0_1_XX + H_0_1_YY + H_0_1_ZZ)
        H_0_2 = J_0_2 * (H_0_2_XX + H_0_2_YY + H_0_2_ZZ)
        H_0_3 = J_0_3 * (H_0_3_XX + H_0_3_YY + H_0_3_ZZ)
        H_1_2 = J_1_2 * (H_1_2_XX + H_1_2_YY + H_1_2_ZZ)
        H_1_3 = J_1_3 * (H_1_3_XX + H_1_3_YY + H_1_3_ZZ)
        H_2_3 = J_2_3 * (H_2_3_XX + H_2_3_YY + H_2_3_ZZ)
        
        H = H_0_1 + H_0_2 + H_0_3 + H_1_2 + H_1_3 + H_2_3
        eigenvalues = np.sort(eigh(H)[0])
        m_eigenvalues = -eigenvalues
        m_eigenvalues = np.sort(m_eigenvalues)
        spectrums.append(eigenvalues)
        m_spectrums.append(m_eigenvalues)
        
    return spectrums, m_spectrums

def check_symmetry(spectrums, m_spectrums):
    symmetry = np.array(np.zeros(64))
    eigenvalues = []
    idx = 0
    
    #check_symmetry
    for i in range(64):
        if i == 0:
            idx = 1
            symmetry[i] = idx
            eigenvalues.append(spectrums[i])
            continue
        for j in range(i):
            isdiff = True
            if np.allclose(spectrums[i], spectrums[j], atol=0.01):
                isdiff = False
                break
            elif np.allclose(spectrums[i], m_spectrums[j], atol=0.01):
                isdiff = False
                break
        if not isdiff:
            symmetry[i] = symmetry[j]
        else:
            idx += 1
            symmetry[i] = idx
            eigenvalues.append(spectrums[i])
    
    Jij_set = []
    comb = 1
    
    #cal J_ij
    for num in range(64):
        if comb > idx:
            break
        if comb == symmetry[num]:
            binary_string = bin(num)[2:].zfill(6)
            comb += 1
            Jij = []
            
            J_0_1 = 2 * int(binary_string[5]) - 1
            J_0_2 = 2 * int(binary_string[4]) - 1
            J_0_3 = 2 * int(binary_string[3]) - 1
            J_1_2 = 2 * int(binary_string[2]) - 1
            J_1_3 = 2 * int(binary_string[1]) - 1
            J_2_3 = 2 * int(binary_string[0]) - 1
            
            Jij.append(J_0_1)
            Jij.append(J_0_2)
            Jij.append(J_0_3)
            Jij.append(J_1_2)
            Jij.append(J_1_3)
            Jij.append(J_2_3)
            
            Jij_set.append(Jij)
        
    return idx, eigenvalues, Jij_set

if __name__=='__main__':
    spectrums, m_spectrums = generate_hamiltonian_eigenvalue()
    num_symmetry_classes, eigenvalues, Jij_set = check_symmetry(spectrums, m_spectrums)
    print('number of symmetry classes : ' + str(num_symmetry_classes))
    
    for i in range(num_symmetry_classes):
        print(str(i + 1) + ' sysmmetry class')
        print('J01 : ' + str(Jij_set[i][0]))
        print('J02 : ' + str(Jij_set[i][1]))
        print('J03 : ' + str(Jij_set[i][2]))
        print('J12 : ' + str(Jij_set[i][3]))
        print('J13 : ' + str(Jij_set[i][4]))
        print('J23 : ' + str(Jij_set[i][5]))
        print('eigenspectrums : ' + str(eigenvalues[i]))
        print('')
        print('')