# Ferromagnetic Ring Simulation

This repository contains a Python simulation of the rotational dynamics of two concentric ferromagnetic objects (rings or disks) suspended by vertical strings and influenced by an external magnetic field from an overhead magnet.

## Description

The simulation models the interplay between mechanical torsional springs and magnetic interactions that cause the objects to rotate. The equations of motion are derived using the Lagrangian formalism, and the simulation numerically solves the full nonlinear differential equations without relying on the small angle approximation.

## Features

- **Customizable Parameters**: Adjust torsional spring constants, magnetic interaction strength, object dimensions, densities, and initial conditions.
- **Support for Rings and Disks**: Simulate the system with the inner object being either a ring or a disk.
- **Visualization**: Animated visualization of the rotational motion, along with plots of kinetic and potential energies over time.
- **Energy Conservation Analysis**: Tracks total energy and energy drift to assess numerical stability and accuracy.
- **Parallel Simulations**: Ability to run multiple simulations in parallel with varying parameters.
