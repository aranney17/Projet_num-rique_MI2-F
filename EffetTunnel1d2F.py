import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded


hbar = 1.054571817e-34
m = 9.1093837139e-31



#evolution numerique (CRANK-NICOLSON)

def cn_evolution(psi0, V, x, dt, nsteps, record_every):
    """
    Evolution de la fonction d'onde dans le temps.
    Retourne les instants et les états du système.
    """
  
    nx = len(x)
    dx = x[1] - x[0]

    r = 1j*hbar*dt/(4*m*dx**2)
    s = 1j*dt/(2*hbar)
    ab = np.zeros((3, nx), complex)
    ab[0,1:] = -r
    ab[1,:] = 1 + 2*r + s*V
    ab[2,:-1] = -r

    psi = psi0.astype(complex).copy()

    snaps = [psi.copy()]
    times = [0.0]

    for n in range(nsteps):

        d = (1 - 2*r - s*V)*psi
        d[1:] += r*psi[:-1]
        d[:-1] += r*psi[1:]

        psi = solve_banded((1,1), ab, d)

        psi[0] = 0
        psi[-1] = 0

        if (n+1) % record_every == 0:
            snaps.append(psi.copy())
            times.append((n+1)*dt)

    return np.array(times), snaps, dx



#paquet d'onde initial
def paquet_initial(x, x0, k0, a_pkt, dx):
    """Paquet d'onde gaussien normalisé"""
    psi = (2/(np.pi*a_pkt**2))**0.25 \
          * np.exp(1j*k0*(x-x0)) \
          * np.exp(-(x-x0)**2/a_pkt**2)

    return psi / np.sqrt(np.sum(np.abs(psi)**2)*dx)



#temp de traversee
def temps_traversee(times, snaps, x, a_bar, dx):
    """
    Temps quand la probabilité transmise atteint la moitié
    de sa valeur finale
    """

    Tt = np.array([
        np.sum(np.abs(s[x > a_bar])**2)*dx for s in snaps
    ])

    T_final = Tt[-1]

    if T_final < 1e-6:
        return np.nan, Tt, T_final

    seuil = T_final / 2
    idx = np.argmax(Tt >= seuil)

    return times[idx], Tt, T_final



#PROGRAMME PRINCIPAL
if __name__ == "__main__":

    # paramètres du paquet
    a_pkt = 5e-9
    k0 = 8e9

    #energie moyenne
    E = hbar**2 * k0**2 / (2*m)

    #barrière de potentiel
    a_bar = 0.5e-9
    V0 = 1.2 * E

    #domaine spatial
    x = np.linspace(-5e-8, 5e-8, 3000)

    # potentiel barrière
    V = np.where((x > 0) & (x < a_bar), V0, 0.0)

    # paquet initial
    x0 = -2.0e-8
    dt = 2e-18
    nsteps = 27000
    rec = 600

    psi0 = paquet_initial(x, x0, k0, a_pkt, dx=1)

    # évolution temporelle
    times, snaps, dx = cn_evolution(psi0, V, x, dt, nsteps, rec)


    
    #graphique visuel de la propagation
    fig, ax = plt.subplots()

    for i in [0, len(snaps)//3, 2*len(snaps)//3, -1]:
        ax.plot(x*1e9, np.abs(snaps[i])**2,
                label=f"t={times[i]*1e15:.0f} fs")

    ax.axvspan(0, a_bar*1e9, color="grey", alpha=0.3, label="barrière")

    ax.set_xlabel("x(nm)")
    ax.set_ylabel("|ψ|^2")
    ax.set_title("Effet tunnel")
    ax.legend()
    plt.show()


  
    #transmision et reflexion
    psiF = snaps[-1]

    T_num = np.sum(np.abs(psiF[x > a_bar])**2)*dx
    R_num = np.sum(np.abs(psiF[x < 0])**2)*dx
    print("E/V0 =", E/V0)
    print("Transmission =", T_num)
    print("Reflexion =", R_num)


   
    # temps de traversee
    vg = hbar * k0 / m
    tau0_th = a_bar / vg

    print("Temps libre =", tau0_th)

    tau_t, _, _ = temps_traversee(times, snaps, x, a_bar, dx)
    print("Temps tunnel =", tau_t)


    #etude parametre
    print("\nInfluence de a :")

    for ab_ in [0.3e-9, 0.5e-9, 0.8e-9]:

        Vb = np.where((x > 0) & (x < ab_), V0, 0.0)
        _, sn, _ = cn_evolution(psi0, Vb, x, dt, nsteps, nsteps)

        Tn = np.sum(np.abs(sn[-1][x > ab_])**2)*dx
        print("a =", ab_, "-> T =", Tn)


    print("\nInfluence de V0 :")

    for fac in [0.8, 1.2, 1.5]:

        V0_ = fac * E
        Vb = np.where((x > 0) & (x < a_bar), V0_, 0.0)

        _, sn, _ = cn_evolution(psi0, Vb, x, dt, nsteps, nsteps)
      
        Tn = np.sum(np.abs(sn[-1][x > a_bar])**2)*dx
        print("V0/E =", fac, "-> T =", Tn)
