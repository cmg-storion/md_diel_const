# Dielectric Constant Calculator

## Contents

* [About](#about)
* [Installation](#installation)
* [Examples](#examples)

### About

This repository provides a Python workflow for calculating dielectric constants of molecular liquids from molecular dynamics simulations using MACE-POLAR.

The package provides:

* MACE-POLAR molecular dynamics
* extraction of atomic charges and dipoles
* trajectory conversion to DCD format
* dielectric constant calculation from dipole fluctuations
* dielectric constant convergence analysis

### Installation

```bash
git clone https://github.com/cmg-storion/md_diel_const
cd md_diel_const
pip install -r requirements.txt
```

### Examples

Example workflows and notebooks are provided in the `examples/` directory.

The repository includes an EC/EMC example demonstrating the complete workflow from system preparation and MACE-POLAR molecular dynamics to dielectric constant calculation.

