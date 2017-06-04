import scipy.integrate as scp
import numpy as np
import matplotlib.pyplot as plt


# Variables paramètre
M_0 = (9300e3, 0)
v_0 = (0, 3000)
M = 6.0e24
m = 1e3

# Constantes
G = 6.7e-11
K1 = M * G
K2 = m * abs(M_0[0] * v_0[1] - M_0[1] * v_0[0])


# Fonction de dérivation de l'équation différentielle
def f(R, _):
    r, rp = R
    Rp = [rp, (((K2 / m)**2) - r * K1) / (r**3)]
    return Rp


# Initialisation des données de calcul
t = np.linspace(0, 12000, 10000)
dt = t[1] - t[0]
R0 = np.array([np.sqrt(M_0[1]**2 + M_0[0]**2), M_0[0] * v_0[0] + M_0[1] * v_0[1]])

# Calcul de r et theta
rs = [e[0] for e in scp.odeint(f, R0, t)]
thetas_p = [K2 / (m * e**2) for e in rs]
thetas = [0.]
for i in range(1, len(rs)):
    thetas.append(thetas[-1] + dt * (thetas_p[i] + thetas_p[i-1])/2)

# Changement de base
x = [rs[i] * np.cos(thetas[i]) for i in range(len(rs))]
y = [rs[i] * np.sin(thetas[i]) for i in range(len(rs))]


# Affichage des courbes
traj, = plt.plot(x, y)
astre, = plt.plot([0], "ro", markersize=10)

max_total = max(max(x), max(y), -min(x), -min(y))

plt.xlabel('x')
plt.ylabel('y')
plt.legend([traj, astre], ['Trajectoire', 'Astre central'], loc=9)
plt.axis([-1.1 * max_total, 1.1 * max_total] * 2)

plt.show()
