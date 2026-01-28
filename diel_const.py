import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda

def dielectric_constant(topology, trajectory, dump, ts = 1.0, T = 298.0, save_plot = "diel_const_convergency.png"):
    '''
    Calculation of dielectric constant by data from MD simulation
    
    topology (str): file with system topology; must contain partial charges
    trajectory (str): file with trajectory in 'dcd' format
    dump (int): trajectory recording frequency
    ts (float): timestep of simulation in fs
    T (float): simulation temperature in 'K'
    save_plot (str): path to save plot 

    return: 
    '''

    #Constants
    const = 698709 # convertion to SI
    t_start_ns = 5.0

    u = mda.Universe(topology, trajectory)
    atoms = u.atoms
    charges = atoms.charges
    dt_ns = (dump * ts) * 1e-6

    dipoles = []
    volumes = []
    times = []  

    for i, s in enumerate(u.trajectory):
        r = atoms.positions          # Å
        M = np.sum(charges[:, None] * r, axis=0)  # e·Å
        dipoles.append(M)
        volumes.append(s.volume)    # Å^3
        times.append(i*dt_ns)    

    dipoles = np.array(dipoles)
    volumes = np.array(volumes)
    times = np.array(times)

    
    # Dielectric convergence
    M2 = np.sum(dipoles**2, axis=1)

    eps_t = []
    for i in range(1, len(M2) + 1):
        eps = 1.0 + (const / (np.mean(volumes[:i]) * T)) * np.mean(M2[:i])
        eps_t.append(eps)
    eps_t = np.array(eps_t)

    # Plateau averaging
    mask = times >= t_start_ns
    eps_mean = np.mean(eps_t[mask])
    eps_std = np.std(eps_t[mask])
    print(f"Dielectric constant ε = {eps_mean:.2f} ± {eps_std:.2f}")


    # Plot
    y_min = max(0, eps_mean - 20)
    y_max = eps_mean + 20

    plt.figure(figsize=(7,5))
    plt.plot(times, eps_t, lw=2, label="ε(t)")
    plt.axhline(eps_mean, ls="--", color = 'orange', lw=2, label=f"⟨ε⟩ = {eps_mean:.2f}")
    plt.xlabel("Time (ns)", fontsize=12)
    plt.ylabel("ε", fontsize=12)
    plt.ylim(y_min, y_max)
    plt.legend()
    plt.tight_layout()

    if save_plot:
        plt.savefig(save_plot)
        print(f"The convergency plot has been saved to file:  {save_plot}")

    plt.show()
    return eps_t, eps_mean



if __name__ == "__main__":
    top = "post_npt.data"
    traj = "prod_pc.dcd"    
    dump = 1000    

    # Вызов функции
    history, result = dielectric_constant(top, traj, dump, save_plot = None)
