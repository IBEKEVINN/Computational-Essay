import numpy as np
import matplotlib.pyplot as plt

#constants
hbar=1.055e-34 #Js
q=1.602e-19    #C
m=9.1e-31      #kg
effective_m = 0.2*m

#grid
Np=100
a=1e-10    #m
X=a*np.linspace(1, Np, Np)/1e-9  #nm
mid = (Np/2)*a/1e-9

#Define Hamiltonian as a tridiagonal matrix
t0=(hbar*hbar)/(2*effective_m*a*a)/q #divide by q to convert to eV
on=2.0*t0*np.ones(Np)
off=-t0*np.ones(Np-1)

# Potential for 2 QD
# 1st Scenario (Weak Interaction)

alpha = 0.5

Energy_gap_array = []

seperation_distance = np.linspace(2,9,12)

for distance in seperation_distance:
    dot1 = mid-(distance/2)
    dot2 = mid+(distance/2)
    
    U = []
    for value in X:
        U.append(min(alpha*(value-dot1)**2, alpha*(value-dot2)**2))
        
    H=np.diag(on+U)+np.diag(off,1)+np.diag(off,-1)
    W,V=np.linalg.eig(H)
    idx = W.argsort()[::1]   
    W = W[idx]
    V = V[:,idx]
    Energy_gap=W[1]-W[0]
    Energy_gap_array.append(Energy_gap)

plt.figure(1)
plt.plot(seperation_distance, Energy_gap_array)
plt.xlabel('Distance (nm)')
plt.ylabel('Potential Energy (eV)')
plt.title('Test')
plt.show()

