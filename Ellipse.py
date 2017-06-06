import scipy.integrate as scp
import numpy as np
import matplotlib.pyplot as plt
import time


# Variables paramètre
M_0 = (9300e3, 12000e3)
v_0 = (0, 3000)
M = 6.0e24

# Constantes
G = 6.7e-11
K1 = M * G
K2 = abs(M_0[0] * v_0[1] - M_0[1] * v_0[0])


# Fonction de dérivation de l'équation différentielle
def f(R, _):
    r, rp = R
    Rp = np.array([rp, (((K2)**2) - r * K1) / (r**3)])
    return Rp


def module(x):
    return np.sqrt(x[0] ** 2 + x[1] ** 2)


def prod_scalaire(x, y):
    return x[0] * y[0] + x[1] * y[1]


def rk1(R, h):
    return R + h * f(R, 0)


def rk2(R, h):
    k = R + f(R, 0) * h / 2
    return R + f(k, 0) * h / 2


def rk4(R, h):
    k1 = f(R, 0)
    k2 = f(R + k1 * (h / 2), 0)
    k3 = f(R + k2 * (h / 2), 0)
    k4 = f(R + k3 * h, 0)
    return R + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


# Initialisation des données de calcul
dt = 10
R0 = np.array([module(M_0), prod_scalaire(M_0, v_0) / module(M_0)])

# # Calcul de r et theta
# rs = [e[0] for e in scp.odeint(f, R0, t)]
# thetas_p = [K2 / (m * e**2) for e in rs]
# thetas = [0.]
# for i in range(1, len(rs)):
#     thetas.append(thetas[-1] + dt * (thetas_p[i] + thetas_p[i-1])/2)
#
# # Changement de base
# x = [rs[i] * np.cos(thetas[i]) for i in range(len(rs))]
# y = [rs[i] * np.sin(thetas[i]) for i in range(len(rs))]

T0 = time.time()

Rs = [R0]
thetas = []
if M_0[1] > 0:
    thetas.append(np.arccos(M_0[0] / module(M_0)))
else:
    thetas.append(-np.arccos(M_0[0] / module(M_0)))
a = M_0[1] / M_0[0]
x, y = [M_0[0]], [M_0[1]]


def continuer():
    Rs.append(rk4(Rs[-1], dt))
    thetas.append(thetas[-1] + dt * K2 / (Rs[-1][0]**2))
    x.append(Rs[-1][0] * np.cos(thetas[-1]))
    y.append(Rs[-1][0] * np.sin(thetas[-1]))

T1 = time.time()

while x[-1] * a <= y[-1]:
    continuer()

while x[-1] * a > y[-1]:
    continuer()

x.pop()
y.pop()

T2 = time.time()
print(T2 - T1)

# x0, y0 = np.array(M_0) / module(M_0)
# M = np.array([[x0**2 - y0**2, 2 * x0 * y0], [2 * x0 * y0, y0**2 - x0**2]])


# def sym(A):
#     u, v = A
#     return [u * M[0, 0] + v * M[0, 1], u * M[1, 0] + v * M[1, 1]]
#
#
# for i in range(len(x) - 1, 0, -1):
#     x_p, y_p = sym([x[i], y[i]])
#     x.append(x_p)
#     y.append(y_p)
#
# x.append(x[0])
# y.append(y[0])

Tf = time.time()

print(Tf - T0)


# Affichage des courbes
traj, = plt.plot(x, y)
astre, = plt.plot([0], "ro", markersize=10)

max_total = max(max(x), max(y), -min(x), -min(y))

plt.xlabel('x')
plt.ylabel('y')
plt.legend([traj, astre], ['Trajectoire', 'Astre central'], loc=9)
plt.axis([-1.1 * max_total, 1.1 * max_total] * 2)

plt.show()
