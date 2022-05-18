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

# Potential for 2 QD
# 1st Scenario (Weak Interaction)
dot1 = 4
dot2 = 6
dotDistance = dot2-dot1
alpha = 0.5
U = []
for value in X:
    U.append(min(alpha*(value-dot1)**2, alpha*(value-dot2)**2))

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

#calculate probablity
Psi0 = V[:,0]
Psi1 = V[:,1]

Psi0_prob=np.multiply(V[:,3],V[:,3])
Psi1_prob=np.multiply(V[:,2],V[:,2])

plt.figure(1)
plt.plot(X, U)
plt.xlabel('Distance (nm)')
plt.ylabel('Potential Energy (eV)')
plt.title('DQD Potential Energy at $\\alpha$ = %.1f and R = %d' % (alpha,dotDistance))
plt.show()

plt.figure(2)
plt.plot(X, Psi0)
plt.xlabel('Distance (nm)')
plt.title('Psi_0 Wavefunction')
plt.show()

plt.figure(3)
plt.plot(X, Psi0_prob)
plt.xlabel('Distance (nm)')
plt.ylabel('Probability')
plt.title('|Psi_0|^2')
plt.show()

plt.figure(4)
plt.plot(X, Psi1)
plt.xlabel('Distance (nm)')
plt.title('Psi_1 Wavefunction')
plt.show()

plt.figure(4)
plt.plot(X, Psi1_prob)
plt.xlabel('Distance (nm)')
plt.ylabel('Probability')
plt.title('|Psi_1|^2')
plt.show()

print(W[1]-W[0])