import os
import glob
import time
import numpy as np

from ase import units
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md import MDLogger
from ase.io.trajectory import Trajectory
from ase.constraints import FixCom
from ase.optimize import BFGS

from mace.calculators import mace_polar


def run_mace_polar_md(geometry_file, T=300, equilibration_steps=50000, production_steps=2000000, output_prefix="mace_polar_md",):
    """
    Run MD for dielectric constant simulation in NVT and calculation of charges and atomic dipoles

    Parameters:
    geometry_file : str
        Path to the input geometry file containing the pre-packed
        molecular system, e.g. PDB or XYZ.

    T : float
        MD simulation temperature in K.
        Default: 300 K.

    equilibration_steps : int
        Number of MD steps used for equilibration..
        Default: 50,000.

    production_steps : int
        Number of MD steps used for production.
        Default: 2,000,000.

    output_prefix : str
        Prefix used for all output files in calculation.
        Existing files starting with this prefix are removed before the calculation starts.
        Default: "mace_polar_md".

    Notes
    -----
    The simulation consists of:

    1. Reading the pre-packed molecular system.
    2. Setting periodic boundary conditions.
    3. Geometry optimization using BFGS.
    4. Maxwell-Boltzmann velocity initialization.
    5. Langevin MD equilibration.
    6. Langevin MD production run.
    7. Saving the trajectory, energies/logs, final geometry,
       atomic charges, and atomic dipoles.

    The MACE-POLAR-1-m model is used with CUDA and float32.

    !!! All existing files whose names start with `output_prefix`
    are deleted before the calculation starts.
    """


    # REMOVE OLD OUTPUT FILES

    old_files = glob.glob(f"{output_prefix}*")

    if old_files:
        print(f"\nRemoving existing files with prefix '{output_prefix}'")

        for file in old_files:
            if os.path.isfile(file):
                os.remove(file)


    # LOAD SYSTEM

    atoms = read(geometry_file)
    cell = atoms.cell
    a, b, c = cell.lengths()

    atoms.set_cell([a, b, c])
    atoms.set_pbc([True, True, True])
    atoms.center()

    # Keep the center of mass fixed
    atoms.set_constraint(FixCom())

    print(f"Number of atoms: {len(atoms)}")
    print(f"Temperature: {T} K")
    print(f"Cell: {a:.3f} x {b:.3f} x {c:.3f} Å")


    # CALCULATOR PARAMETERS

    print("\nINITIALIZING MACE-POLAR")

    calc = mace_polar(
        model="polar-1-m",
        device="cuda",
        default_dtype="float32"
    )

    atoms.info["charge"] = 0
    atoms.info["spin"] = 1
    atoms.info["external_field"] = [0.0, 0.0, 0.0]

    atoms.calc = calc


    # GEOMETRY OPTIMIZATION

    print("\nGEOMETRY OPTIMIZATION")

    opt = BFGS(atoms, ogfile=f"{output_prefix}_e_min.log")
    opt.run(fmax=0.05)
    max_force = max(abs(atoms.get_forces().flatten()))

    print(f"Max force: {max_force:.6f} eV/Å")

    write(f"{output_prefix}_e_min.xyz",atoms)

    print("GEOMETRY OPTIMIZATION DONE")

    # INITIAL VELOCITIES

    MaxwellBoltzmannDistribution(atoms, temperature_K=T)
    Stationary(atoms)
    kinetic_energy = atoms.get_kinetic_energy()

    # MD SETUP

    dyn = Langevin(atoms,
                    timestep=1 * units.fs,
                    temperature_K=T,
                    friction=0.01,
                    fixcm=False
                    )


    chrg_list = []
    at_dip_list = []

    # LOG MULTIPOLES

    def log():
        res = atoms.calc.results

        if "density_coefficients" not in res:
            raise RuntimeError(
                "density_coefficients not found — "
                "model does not provide multipoles"
            )

        dc = res["density_coefficients"]

        # Charge
        charges = dc[:, 0]

        # Atomic dipoles

        at_dipoles = dc[:, [3, 1, 2]]

        chrg_list.append(charges.copy())

        at_dip_list.append(at_dipoles.copy())


    # REMOVE CENTER-OF-MASS DRIFT
    def rm_drift():
        Stationary(atoms)


    # EQUILIBRATION
    print("\nEQUILIBRATION")

    dyn.attach(rm_drift, interval=1000)

    dyn.attach(MDLogger(dyn, atoms, f"{output_prefix}_equilibration.log", header=True), interval=1000)

    dyn.run(equilibration_steps)

    print("EQUILIBRATION DONE")
    dyn.observers.clear()
    write(f"{output_prefix}_relaxed.xyz",atoms)


    # PRODUCTION
    print("\nPRODUCTION")

    traj = Trajectory(f"{output_prefix}_production.traj", "w", atoms)

    dyn.attach(rm_drift, interval=1000)

    dyn.attach(traj.write,interval=1000)

    # Save atomic charges and dipoles
    dyn.attach(log, interval=1000)

    dyn.attach(MDLogger(dyn, atoms, f"{output_prefix}_production.log", header=True), interval=1000)


    start = time.perf_counter()

    dyn.run(production_steps)

    end = time.perf_counter()

    traj.close()

    # FINAL GEOMETRY
    write(f"{output_prefix}_final_geometry.xyz", atoms)
    print("\nPRODUCTION DONE")
    print(f"Total production run time: "
        f"{end - start:.2f} seconds")

    # SAVE CHARGES AND DIPOLES

    np.save(f"{output_prefix}_charges.npy", np.array(chrg_list))

    np.save(f"{output_prefix}_atomic_dipoles.npy", np.array(at_dip_list))

    print("\nCHARGES AND ATOMIC DIPOLES SAVED")

    print(f"Number of saved frames: {len(chrg_list)}")

    print("\nMD CALCULATION COMPLETED")