
import numpy as np
import matplotlib.pyplot as plt

# constantes physiques
hbar = 1.054571817e-34
m = 9.1093837139e-31



# 1) derivee numeriques
def derivee_premiere(f, dx):
    d = np.zeros_like(f)
    d[1:-1] = (f[2:] - f[:-2]) / (2*dx)
    d[0] = (f[1] - f[0]) / dx
    d[-1] = (f[-1] - f[-2]) / dx
    return d


def derivee_seconde(f, dx):
    d = np.zeros_like(f)
    d[1:-1] = (f[2:] - 2*f[1:-1] + f[:-2]) / dx**2
    d[0] = d[1]
    d[-1] = d[-2]
    return d



# 2)paquet d'onde gaussien theorique

def GaussWP(k0, a, x, t):
    alpha = 1 + 2j*hbar*t/(m*a**2)

    return ((2/(np.pi*a**2))**0.25 / np.sqrt(alpha)
            * np.exp(1j*(k0*x - hbar*k0**2*t/(2*m)))
            * np.exp(-(x - hbar*k0*t/m)**2/(a**2*alpha)))



# 3) evolution numerique

def schrodinger_euler(psi0, V, x, dt, nsteps):
    nx = len(x)
    dx = x[1] - x[0]

    psi = psi0.astype(complex).copy()

    coef = 1j * hbar * dt / (2 * m * dx**2)

    for _ in range(nsteps):

        psi_new = np.zeros_like(psi, dtype=complex)

        for i in range(1, nx-1):

            laplacien = psi[i+1] - 2*psi[i] + psi[i-1]

            psi_new[i] = (
                psi[i]
                + coef * laplacien
                - 1j * dt / hbar * V[i] * psi[i]
            )

        # conditions aux bords (puits infini)
        psi_new[0] = 0
        psi_new[-1] = 0
        psi = psi_new

    return psi



# 4) PROGRAMME PRINCIPAL
if __name__ == "__main__":

  
    # test derivee
    x = np.linspace(0, 1, 1000)
    dx = x[1] - x[0]

    f = x**2

    d1 = derivee_premiere(f, dx)
    d2 = derivee_seconde(f, dx)
    print("Erreur dérivée 1 :", np.max(np.abs(d1 - 2*x)))
    print("Erreur dérivée 2 :", np.max(np.abs(d2 - 2)))


    # Parametre physiques
    a = 3e-9
    k0 = 6e9

    x = np.linspace(-4e-8, 4e-8, 2000)
    dx = x[1] - x[0]

    psi0 = GaussWP(k0, a, x, 0.0)


    #evolution num
    T = 3e-15
    dt = 1e-18
    nsteps = int(T / dt)

    psi_num = schrodinger_euler(psi0, np.zeros_like(x), x, dt, nsteps)
    psi_th = GaussWP(k0, a, x, T)


    #norme et erreur
    norme = np.sum(np.abs(psi_num)**2) * dx
    erreur = np.sqrt(np.sum(np.abs(psi_num - psi_th)**2) * dx)

    print("Norme finale :", norme)
    print("Erreur L2 :", erreur)


 #graphique
    plt.plot(x, np.abs(psi0)**2, label="Initial")
    plt.plot(x, np.abs(psi_num)**2, label="Numérique")
    plt.plot(x, np.abs(psi_th)**2, "--", label="Théorique")

    plt.xlabel("x(m)")
    plt.ylabel("|ψ|^2")
    plt.title("Paquet d’onde - Schrodinger 1D")
    plt.legend()
    plt.show()
