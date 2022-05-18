import numpy as np
import matplotlib.pyplot as plt

#constants
hbar=1.055e-34 #Js
q=1.602e-19    #C
m=9.1e-31      #kg

#grid
Np=100
a=1e-10    #m
X=a*np.linspace(1, Np, Np)/1e-9  #nm

#Define Hamiltonian as a tridiagonal matrix
t0=(hbar*hbar)/(2*m*a*a)/q #divide by q to convert to eV
on=2.0*t0*np.ones(Np)
off=-t0*np.ones(Np-1)

#Define 'particle in a box' potential
n1=25
n2=75 
x1=n1*a/1e-9
x2=n2*a/1e-9
print(x1)
print(x2)
V0=0.5; #eV
U=np.array(V0*np.ones(n1))
U=np.append(U,np.zeros(n2-n1))
U=np.append(U,V0*np.ones(Np-n2)) 
print("Potential Array")
print(U)

#Define Hamiltonian
H=np.diag(on+U)+np.diag(off,1)+np.diag(off,-1)
print("Hamiltonian")
print(H)

#solve for eigenvalues and vectors
W,V=np.linalg.eig(H)
idx = W.argsort()[::1]   
W = W[idx]
V = V[:,idx]
print("Eigenvalues and Eigenvectors")
print(W)
print(V)

#print eigenvalues and energy gaps
print("Eigenvalues and Energy Gap")
print(W[0:10])
Energy_gap=W[1]-W[0]
print(Energy_gap)

#calculate probablity
Psi0=np.multiply(V[:,0],V[:,0])
Psi1=np.multiply(V[:,1],V[:,1])

plt.figure(1)
plt.plot(X, U)
plt.xlabel('Distance (nm)')
plt.ylabel('Potential Energy (eV)')
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
