import numpy as np
import matplotlib.pyplot as plt

# constantes physiques
hbar = 1.054571817e-34
m = 9.1093837139e-31


def GaussWP(k0, a, x, t):
    """Paquet d'onde gaussien libre"""
    alpha = 1 + 2j*hbar*t/(m*a**2)
    xc = hbar*k0*t/m

    pref = (2/(np.pi*a**2))**0.25 / np.sqrt(alpha)
    phase = np.exp(1j*(k0*x - hbar*k0**2*t/(2*m)))
    envelope = np.exp(-(x - xc)**2/(a**2*alpha))

    return pref * phase * envelope


if __name__ == "__main__":

    k0 = 5e9
    a = 1e-9

    x = np.linspace(-6e-9, 6e-9, 2000)

    # t=0
    psi = GaussWP(k0, a, x, 0.0)

    plt.figure()
    plt.plot(x, np.real(psi), label="Re(Ψ)")
    plt.plot(x, np.imag(psi), label="Im(Ψ)")
    plt.plot(x, np.abs(psi), "k--", label="|Ψ| (enveloppe)")
    plt.title("Paquet d'onde gaussien (t=0)")
    plt.xlabel("x(m)")
    plt.legend()
    plt.show()

    # difficulté : oscillations rapides
    k0_big = 5e10
    psi_big = GaussWP(k0_big, a, x, 0.0)

    plt.figure()
    plt.plot(x, np.real(psi_big))
    plt.title("Difficulté : Re(Ψ) illisible si k0 grand")
    plt.xlabel("x(m)")
    plt.show()

    # solution : densité de probabilité
    plt.figure()
    plt.plot(x, np.abs(psi_big)**2)
    plt.title(" tracer |Ψ|^2 (plus lisible)")
    plt.xlabel("x(m)")
    plt.show()

  # bonus!
    # évolution temporelle
    plt.figure()
    for t in [0, 2e-15, 4e-15, 6e-15]:
        psi_t = GaussWP(k0, a, x, t)
        plt.plot(x, np.abs(psi_t)**2, label=f"t={t:.0e}s")

    plt.title("Étalement du paquet d'onde")
    plt.xlabel("x(m)")
    plt.ylabel("|Ψ|^2")
    plt.legend()
    plt.show()

    # normalisation
    xn = np.linspace(-3e-8, 3e-8, 400000)
    dx = xn[1] - xn[0]

    for t in [0, 3e-15]:
        norm = np.sum(np.abs(GaussWP(k0, a, xn, t))**2) * dx
        print(f"t={t:.1e} s -> norme = {norm:.6f}")
