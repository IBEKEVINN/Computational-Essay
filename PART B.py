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

#Define Hamiltonian as a tridiagonal matrix
t0=(hbar*hbar)/(2*effective_m*a*a)/q #divide by q to convert to eV
on=2.0*t0*np.ones(Np)
off=-t0*np.ones(Np-1)

# Potential for 1 QD
mid = (Np/2)*a/1e-9

# Selecting alpha of 0.5
alpha = 3.36e-10
U = alpha * (X-mid)**2
alpha_array = np.linspace(0,5e-6, 100)

# Define Hamiltonian
H=np.diag(on+U)+np.diag(off,1)+np.diag(off,-1)

# Solve and find lowest 2 eigenvalues
W,V=np.linalg.eig(H)
idx = W.argsort()[::1]   
W = W[idx]
V = V[:,idx]
print("Eigenvalues and Eigenvectors")
print("Lowest eigenvalue and corresponding wavefunction")
print(W[0])
print(V[0])
print("Seconed lowest eigenvalue and corresponding wavefunction")
print(W[1])
print(V[1])

# Calculate Probability
Psi0=np.multiply(V[:,0],V[:,0])
Psi1=np.multiply(V[:,55],V[:,55])

print((W[1]-W[0])/(10**-3))

plt.figure(1)
plt.plot(X, U)
plt.xlabel('Distance (nm)')
plt.ylabel('Potential Energy (eV)')
plt.title('QD Potential Energy at $\\alpha$ = %.1f' % alpha)
plt.show()

plt.figure(2)
plt.plot(X, Psi0)
plt.xlabel('Distance (nm)')
plt.ylabel('Probability')
plt.title('|Psi_0|^2')
plt.show()

plt.figure(3)
plt.plot(X, Psi1)
plt.xlabel('Distance (nm)')
plt.ylabel('Probability')
plt.title('|Psi_1|^2')
plt.show()

# Creating Energy gap vs alpha plot

Energy_gap_array = []
for value in alpha_array:
    U_array = value * (X-mid)**2
    H=np.diag(on+U_array)+np.diag(off,1)+np.diag(off,-1)
    W,V=np.linalg.eig(H)
    idx = W.argsort()[::1]
    W = W[idx]
    V = V[:,idx]
    Energy_gap=(W[1]-W[0])/(10**(-3))
    Energy_gap_array.append(Energy_gap)

plt.figure(4)
plt.plot(alpha_array, Energy_gap_array)
plt.xlabel('alpha')
plt.ylabel('Energy Gap (meV)')
plt.title('Energy Gap vs Alpha')
plt.show()


