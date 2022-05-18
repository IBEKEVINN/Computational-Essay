dx    = 0.02                       # spatial separation
x     = np.arange(0, 10, dx)       # spatial grid points
# Laplace Operator (Finite Difference)
D2 = scipy.sparse.diags([1, -2, 1], 
                        [-1, 0, 1],
                        shape=(x.size, x.size)) / dx**2

# RHS of Schrodinger Equation
hbar = 1
def psi_t(t, psi):
    return -1j * (- 0.5 * hbar / m * D2.dot(psi) + V / hbar * psi)

dt = 0.005  # time interval for snapshots
t0 = 0.0    # initial time
tf = 1.0    # final time
t_eval = np.arange(t0, tf, dt)  # recorded time shots
# Solve the Initial Value Problem
sol = scipy.integrate.solve_ivp(psi_t, 
                                t_span = [t0, tf],
                                y0 = psi0, 
                                t_eval = t_eval,
                                method="RK23")