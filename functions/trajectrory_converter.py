import MDAnalysis as mda
from MDAnalysis.coordinates.DCD import DCDWriter
from ase.io import read, write


def convert_traj_to_dcd(trajectory, topology, dcd_output):
    """
    Convert an ASE trajectory to a DCD trajectory and create PDB topology file for MDAnalysis.

    Parameters
    trajectory : str
        Path to the ASE trajectory (.traj).

    topology : str
        Output path for the PDB topology file.

    dcd_output : str
        Output path for the DCD trajectory.
    """

    # Read all frames from ASE trajectory
    frames = read(trajectory, index=":")

    # Save the first frame as topology
    write(topology, frames[0])

    # Create MDAnalysis Universe
    u = mda.Universe(topology)

    # Write DCD trajectory
    with DCDWriter(dcd_output, n_atoms=len(u.atoms)) as w:

        for atoms in frames:

            u.atoms.positions = (atoms.get_positions())
            w.write(u.atoms)