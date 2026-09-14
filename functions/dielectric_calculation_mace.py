import numpy as np
import MDAnalysis as mda
import matplotlib.pyplot as plt


def calculator_dielectric_constant_mace(topology, trajectory, charges_npy, dipoles_npy, dump, ts=1.0, T=300.0, save_plot="diel_const_convergency.png", final_fraction=0.20,  plateau_tolerance=0.03,):
    """
    Calculate the dielectric constant from a MACE-POLAR MD trajectory.

    The dielectric constant is calculated from fluctuations of the
    total dipole moment of the simulation box:

        ε = 1 + C / (V T) <(M - <M>)^2>

    where the total dipole moment includes both the charge contribution
    and the atomic dipole contribution obtained from MACE-POLAR.

    Molecular coordinates are unwrapped across periodic boundary
    conditions before calculating the charge contribution to the
    total dipole moment.

    The function generates a convergence curve for the dielectric
    constant. The final dielectric constant is reported as the mean value,
    together with its standard deviation.


    Parameters
    topology : str
        Path to the topology/structure file, e.g. PDB or XYZ.

    trajectory : str
        Path to the MD trajectory file, e.g. DCD or XTC.

    charges_npy : str
        Path to the NumPy file containing atomic charges saved during the MACE-POLAR MD simulation.

    dipoles_npy : str
        Path to the NumPy file containing atomic dipoles saved during the MACE-POLAR MD simulation.

    dump : int
        Number of MD steps between saved trajectory frames.

    ts : float
        MD timestep in femtoseconds.
        Default: 1.0 fs.

    T : float
        Temperature in K.
        Default: 300.0 K.

    save_plot : str or None
        Filename for the dielectric constant convergence plot.
        If None, the plot is not saved.

    final_fraction : float, optional
        Fraction of the final convergence curve used to determine the final plateau level.
        Default: 0.20 (last 20%).

    plateau_tolerance : float, optional
        Relative tolerance used to identify the plateau.
        Default: 0.03 (±3%).

    Returns
    times : numpy.ndarray
        Simulation time corresponding to each saved trajectory frame, in ns.

    eps_t : numpy.ndarray
        Cumulative dielectric constant calculated using all frames from the beginning of the trajectory up to each time point.

    eps_mean : float
        Mean dielectric constant calculated over the identified plateau region.

    eps_std : float
        Standard deviation of the dielectric constant over the plateau region.

    dipoles : numpy.ndarray
        Total dipole moment of the simulation box for every saved trajectory frame.

    Notes
    The conversion constant

        C = 698709

    is used for the conversion from atomic units to SI of dipole moment, volume, temperature.

    Molecules are identified using the `resid` values from the topology.

    """

    # CONSTANT
    const = 698709

    # UNWRAP MOLECULE

    def unwrap_molecule(r_mol, box):
        """
        Unwrap a molecule across periodic boundary conditions.
        """

        r = r_mol.copy()

        for i in range(1, len(r)):

            dr = r[i] - r[i - 1]
            dr -= box * np.round(dr / box)
            r[i] = r[i - 1] + dr

        return r

    # LOAD SYSTEM

    print("\nLOADING TRAJECTORY")

    u = mda.Universe(topology, trajectory)
    atoms = u.atoms

    charges_all = np.load(charges_npy)
    dipoles_all = np.load(dipoles_npy)

    # CHECK NUMBER OF FRAMES
    n_frames = len(u.trajectory)

    if len(charges_all) != n_frames:

        raise ValueError(
            f"Number of charge frames ({len(charges_all)}) "
            f"does not match the number of trajectory frames "
            f"({n_frames})."
        )

    if len(dipoles_all) != n_frames:

        raise ValueError(
            f"Number of dipole frames ({len(dipoles_all)}) "
            f"does not match the number of trajectory frames "
            f"({n_frames})."
        )

    if charges_all.shape[1] != len(atoms):

        raise ValueError(
            f"Number of atoms in charges "
            f"({charges_all.shape[1]}) "
            f"does not match the topology "
            f"({len(atoms)})."
        )

    if dipoles_all.shape[1] != len(atoms):

        raise ValueError(
            f"Number of atoms in dipoles "
            f"({dipoles_all.shape[1]}) "
            f"does not match the topology "
            f"({len(atoms)})."
        )

    print(f"Number of atoms: {len(atoms)}")

    print(f"Number of trajectory frames: {n_frames}")

    # MOLECULE IDENTIFICATION

    molecule_ids = atoms.resids
    unique_molecules = np.unique(molecule_ids)

    # TIME BETWEEN SAVED FRAMES

    # ts   = MD timestep in fs
    # 1 fs = 1e-6 ns

    dt_ns = (dump * ts) * 1e-6


    dipoles = []
    volumes = []
    times = []

    # DIELECTRIC CONSTANT CALCULATION

    print("\nCALCULATING TOTAL DIPOLE MOMENTS")

    for i, s in enumerate(u.trajectory):

        r = atoms.positions

        q = charges_all[i]

        mu = dipoles_all[i]

        box = s.dimensions[:3]

        M = np.zeros(3)

        # SUM OVER MOLECULES
        for mol_id in unique_molecules:

            indices = np.where(molecule_ids == mol_id)[0]
            r_mol = r[indices].copy()

            # Unwrap molecule across PBC
            r_mol = unwrap_molecule(r_mol, box)
            q_mol = q[indices]
            mu_mol = mu[indices]

            # Charge contribution
            charge_dipole = np.sum(q_mol[:, None] * r_mol, axis=0)

            # Atomic dipole contribution
            atomic_dipole = np.sum(mu_mol, axis=0)

            M += (charge_dipole + atomic_dipole)

        dipoles.append(M)

        volumes.append(s.volume)

        times.append(i * dt_ns)

    dipoles = np.array(dipoles)

    volumes = np.array(volumes)

    times = np.array(times)

    print("TOTAL DIPOLE CALCULATION DONE")


    print("\nCALCULATING DIELECTRIC CONSTANT")

    eps_t = []

    for i in range(1, len(dipoles) + 1):

        M_slice = dipoles[:i]

        M_mean = np.mean(M_slice, axis=0)

        M2_mean = np.mean(np.sum(M_slice**2, axis=1))

        fluct = (M2_mean - np.sum(M_mean**2))

        eps = (1.0 + (const / (np.mean(volumes[:i]) * T)) * fluct)

        eps_t.append(eps)

    eps_t = np.array(eps_t)

    # PLATEAU
    n = len(eps_t)
    n_final = max(10, int(n * final_fraction))

    # Last fraction of the convergence curve
    final_region = eps_t[-n_final:]

    # Median value of the final region
    final_level = np.median(final_region)

    tolerance = (plateau_tolerance * abs(final_level))
    lower_limit = (final_level - tolerance)
    upper_limit = (final_level + tolerance)

    plateau_start_index = None

    for i in range(n):

        remaining = eps_t[i:]

        inside = ((remaining >= lower_limit) & (remaining <= upper_limit))

        fraction_inside = np.mean(inside)

        if fraction_inside >= 0.95:
            plateau_start_index = i
            break

    # If no plateau was detected
    if plateau_start_index is None:
        print(
            "WARNING: No stable plateau detected. "
            "Using the last "
            f"{final_fraction * 100:.0f}% "
            "of the trajectory."
        )

        plateau_start_index = (n - n_final)

    plateau_start_time = (times[plateau_start_index])

    # AVERAGE ε OVER PLATEAU

    eps_plateau = (eps_t[plateau_start_index:])

    eps_mean = np.mean(eps_plateau)

    eps_std = np.std(eps_plateau, ddof=1)

    # PRINT RESULTS
    print("")
    print("DIELECTRIC CONSTANT")
    print("")

    print(
        f"Mean dielectric constant: "
        f"ε = {eps_mean:.2f}"
    )

    print(
        f"Standard deviation: "
        f"± {eps_std:.2f}"
    )

    # PLOT

    plt.figure(figsize=(7, 5))

    plt.plot(times, eps_t, lw=2)
    plt.axhline(eps_mean, lw=2, ls="--", color="orange", label=f"ε = {eps_mean:.2f}")
    plt.xlabel("Time (ns)", fontsize=12)
    plt.ylabel("ε", fontsize=16)
    plt.legend()
    plt.tight_layout()

    plt.xlim(0, times[-1])
    plt.ylim(0, eps_mean + 5)


    if save_plot:

        plt.savefig(save_plot, dpi=300)

        print(
            f"Convergence plot saved to: "
            f"{save_plot}"
        )

    plt.show()


    return (times, eps_t, eps_mean, eps_std, dipoles)