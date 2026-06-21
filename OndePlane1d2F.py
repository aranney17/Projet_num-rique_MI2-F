import numpy as np
from numpy import pi, exp, real, imag, linspace, cos
import matplotlib.pyplot as plt

# constantes physiques
hbar = 1.054571817e-34
m = 9.1093837139e-31

# onde plane 1D
def PlaneWave(amp, k, omega, x, t):
    return amp * exp(1j * (k*x - omega*t))

# relation de dispersion (particule libre)
def omega_libre(k):
    return hbar * k**2 / (2*m)


if __name__ == "__main__":

   
    # 1) onde plane simple
    amp = 1.0
    k = 1.0e10

    x = linspace(-2e-9, 2e-9, 1000)
    omega = omega_libre(k)

    psi = PlaneWave(amp, k, omega, x, 0.0)

    fig, ax = plt.subplots()
    ax.plot(x, real(psi), label="Re(Ψ)")
    ax.plot(x, imag(psi), label="Im(Ψ)")
    ax.set_title("Onde plane 1D")
    ax.set_xlabel("x(m)")
    ax.set_ylabel("Ψ(x,t)")
    ax.legend()
    plt.show()


   
    # 2) superposition de 3 ondes planes

    k0 = 1.0e10
    dk = 1.0e9

    xs = linspace(-pi/dk, pi/dk, 2000)

    o1 = PlaneWave(amp,   k0,      omega_libre(k0),      xs, 0.0)
    o2 = PlaneWave(amp/2, k0-dk/2, omega_libre(k0-dk/2), xs, 0.0)
    o3 = PlaneWave(amp/2, k0+dk/2, omega_libre(k0+dk/2), xs, 0.0)

    somme = o1 + o2 + o3

    enveloppe = amp * (1 + cos(dk * xs / 2))

    fig, ax = plt.subplots()

    ax.plot(xs, real(o1), alpha=0.3, label="k0")
    ax.plot(xs, real(o2), alpha=0.3, label="k0 - dk/2")
    ax.plot(xs, real(o3), alpha=0.3, label="k0 + dk/2")

    ax.plot(xs, real(somme), color="black", lw=1.5, label="Somme")

    ax.plot(xs, enveloppe, "r--", lw=1, label="Enveloppe")
    ax.plot(xs, -enveloppe, "r--", lw=1)

    ax.set_title("Superposition de 3 ondes planes")
    ax.set_xlabel("x(m)")
    ax.set_ylabel("Re(Ψ)")
    ax.legend()

    plt.show()


   
    # 3) Densite de probabilité
  
    fig, ax = plt.subplots()

    ax.plot(xs, np.abs(somme)**2)

    ax.set_title("Densité de probabilité |Ψ|^2")
    ax.set_xlabel("x(m)")
    ax.set_ylabel("|Ψ|^2")

    plt.show()
