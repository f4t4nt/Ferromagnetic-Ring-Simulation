import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend suitable for multiprocessing

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import time
import os
import multiprocessing

def simulate_coupled_rings(
    k=1.0,  # Torsional spring constant (N·m/rad)
    U_rr=1.0,  # Magnetic interaction potential coefficient (N·m)
    rho=7800.0,  # Density of the material (kg/m³), typical for steel
    R1=0.15,  # Radius of the outer ring (m), max 15 cm
    R2=0.1,  # Radius of the inner ring or disk (m)
    r=0.01,  # Cross-sectional radius of the rings or half-thickness of the disk (m)
    is_disk=False,  # If True, inner object is a disk; if False, a ring
    theta1_0=0.0,  # Initial angle of the outer ring (rad)
    omega1_0=0.0,  # Initial angular velocity of the outer ring (rad/s)
    theta2_0=0.0,  # Initial angle of the inner ring or disk (rad)
    omega2_0=0.0,  # Initial angular velocity of the inner ring or disk (rad/s)
    dt=0.001,  # Time step for the simulation (s)
    max_history=None,  # Set to None to retain full history
    save_filename=None,  # Filename to save the animation
    data_filename=None,  # Filename to save the data
    frame_limit=None,  # Number of frames to simulate
    dynamic_lims=True,  # Whether to adjust axes limits dynamically
    fps_video=None  # Frames per second for the saved video; if None, set to 1/dt
):
    # Check for valid parameters
    if (save_filename is not None or data_filename is not None) and frame_limit is None:
        raise ValueError("frame_limit must be specified if save_filename or data_filename is specified.")

    # Calculate fps_video if not provided
    if fps_video is None:
        fps_video = int(1 / dt)
        print(f"fps_video set to {fps_video} based on dt = {dt}")

    # Line lengths for visualization (m)
    line_length1 = 2 * R1
    line_length2 = 2 * R2

    # Simulation state class
    class SimulationState:
        def __init__(self):
            self.theta1 = theta1_0
            self.omega1 = omega1_0
            self.theta2 = theta2_0
            self.omega2 = omega2_0
            self.theta1_prev = self.theta1 - self.omega1 * dt
            self.theta2_prev = self.theta2 - self.omega2 * dt
            self.energies = []
            self.drifts = []
            self.angles = []
            self.omegas = []
            self.time = 0.0
            self.times = []
            self.initial_total_energy = None

    state = SimulationState()

    # Moments of inertia
    I1 = np.pi**2 * R1**3 * r**2 * rho
    if is_disk:
        # Disk
        I2 = 0.5 * np.pi * R2**4 * (2 * r) * rho
        inner_object_description = "Disk"
    else:
        # Ring
        I2 = np.pi**2 * R2**3 * r**2 * rho
        inner_object_description = "Ring"

    # Angular acceleration functions
    def angular_acceleration1(theta1, theta2):
        return (-2 * k * theta1 + k * theta2 + 2 * U_rr * np.sin(2 * (theta1 - theta2))) / I1

    def angular_acceleration2(theta1, theta2):
        return (-k * theta2 + k * theta1 - 2 * U_rr * np.sin(2 * (theta1 - theta2))) / I2

    # Energy functions
    def kinetic_energy_outer(omega1):
        return 0.5 * I1 * omega1**2

    def kinetic_energy_inner(omega2):
        return 0.5 * I2 * omega2**2

    def potential_energy_outer(theta1):
        return 0.5 * k * theta1**2

    def potential_energy_coupling(theta1, theta2):
        return 0.5 * k * (theta1 - theta2)**2

    def potential_energy_ring_ring(theta1, theta2):
        return U_rr * np.cos(2 * (theta1 - theta2))

    # Set matplotlib style
    plt.style.use('ggplot')

    # Create figure and axes
    fig = plt.figure(figsize=(14, 10))
    gs_main = fig.add_gridspec(2, 2)
    fig.suptitle("Coupled Ring Dynamics and Energy")

    # Nested GridSpec
    gs_top_left = gs_main[0, 0].subgridspec(1, 2, width_ratios=[1, 2])

    # Parameters axis
    ax_params = fig.add_subplot(gs_top_left[0, 0])
    ax_params.axis('off')
    param_text = ax_params.text(
        0, 1,
        (
            f"Parameters (SI units):\n"
            f"k = {k} N·m/rad\n"
            f"U_rr = {U_rr} N·m\n"
            f"R1 = {R1} m, R2 = {R2} m\n"
            f"r = {r} m\n"
            f"Density = {rho} kg/m³\n"
            f"dt = {dt} s\n"
            f"Inner object: {inner_object_description}\n\n"
            f"Initial Conditions (SI units):\n"
            f"θ1₀ = {theta1_0} rad, θ2₀ = {theta2_0} rad\n"
            f"ω1₀ = {omega1_0} rad/s, ω2₀ = {omega2_0} rad/s\n"
        ),
        fontsize=10, verticalalignment='top', transform=ax_params.transAxes
    )

    # Ring animation subplot
    ax_ring = fig.add_subplot(gs_top_left[0, 1])
    ax_ring.set_xlim(-line_length1, line_length1)
    ax_ring.set_ylim(-line_length1, line_length1)
    ax_ring.set_aspect('equal', adjustable='box')
    ax_ring.axis("off")
    line2, = ax_ring.plot([], [], color='#ff7f0e', lw=3, label='Inner Object')
    line1, = ax_ring.plot([], [], color='#1f77b4', lw=3, label='Outer Ring')
    ax_ring.legend(loc='upper right')

    # Energy subplot
    ax_energy = fig.add_subplot(gs_main[0, 1])
    ax_energy.set_title("Energy over Time")
    ax_energy.set_xlabel("Time (s)")
    ax_energy.set_ylabel("Energy (J)")
    ax_energy.grid(True)
    colors = plt.get_cmap('tab10')
    energy_lines = {
        "KE Outer": ax_energy.plot([], [], label="Kinetic Energy (Outer)", color=colors(0))[0],
        "KE Inner": ax_energy.plot([], [], label="Kinetic Energy (Inner)", color=colors(1))[0],
        "PE Total": ax_energy.plot([], [], label="Total Potential Energy", color=colors(2))[0],
        "KE Total": ax_energy.plot([], [], label="Total Kinetic Energy", color=colors(3))[0],
        "Total Energy": ax_energy.plot([], [], label="Total Energy", color=colors(4), linestyle="-.", linewidth=2)[0],
    }
    ax_energy.legend(loc='upper left')

    # Angles subplot
    ax_angles = fig.add_subplot(gs_main[1, 0])
    ax_angles.set_title("Angles over Time")
    ax_angles.set_xlabel("Time (s)")
    ax_angles.set_ylabel("Angles (rad)")
    ax_angles.grid(True)
    angle_lines = {
        "Theta1": ax_angles.plot([], [], label="Theta1 (Outer)", color=colors(0))[0],
        "Theta2": ax_angles.plot([], [], label="Theta2 (Inner)", color=colors(1))[0],
    }
    ax_angles.legend(loc='upper left')

    # Energy drift subplot
    ax_drift = fig.add_subplot(gs_main[1, 1])
    ax_drift.set_title("Energy Drift over Time")
    ax_drift.set_xlabel("Time (s)")
    ax_drift.set_ylabel("Energy Drift (J)")
    ax_drift.grid(True)
    drift_line, = ax_drift.plot([], [], label="Energy Drift", color="magenta")
    ax_drift.legend(loc='upper left')

    # Set initial y-limits
    ax_energy.set_ylim(0, 600)
    ax_drift.set_ylim(-0.1, 0.1)
    ax_angles.set_ylim(-4 * np.pi, 4 * np.pi)

    # Adjust layout
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Simulation Parameters
    N_steps_per_frame = 1  # Synchronize simulation time with video time
    print(f"N_steps_per_frame: {N_steps_per_frame}")

    if frame_limit is not None:
        total_frames = frame_limit
        frames = range(total_frames)
    else:
        frames = None  # For indefinite simulation

    # Animation update function
    def update(frame, state):
        # Perform N_steps_per_frame simulation steps
        for _ in range(N_steps_per_frame):
            # Compute accelerations
            alpha1 = angular_acceleration1(state.theta1, state.theta2)
            alpha2 = angular_acceleration2(state.theta1, state.theta2)

            # Verlet integration for θ1
            theta1_new = 2 * state.theta1 - state.theta1_prev + alpha1 * dt**2
            omega1 = (theta1_new - state.theta1_prev) / (2 * dt)

            # Verlet integration for θ2
            theta2_new = 2 * state.theta2 - state.theta2_prev + alpha2 * dt**2
            omega2 = (theta2_new - state.theta2_prev) / (2 * dt)

            # Update positions and velocities
            state.theta1_prev, state.theta1 = state.theta1, theta1_new
            state.theta2_prev, state.theta2 = state.theta2, theta2_new
            state.omega1 = omega1
            state.omega2 = omega2

            # Increment time
            state.time += dt

        # Update line positions
        x2 = [-line_length2 * np.cos(state.theta2), line_length2 * np.cos(state.theta2)]
        y2 = [-line_length2 * np.sin(state.theta2), line_length2 * np.sin(state.theta2)]
        line2.set_data(x2, y2)
        x1 = [-line_length1 * np.cos(state.theta1), line_length1 * np.cos(state.theta1)]
        y1 = [-line_length1 * np.sin(state.theta1), line_length1 * np.sin(state.theta1)]
        line1.set_data(x1, y1)

        # Compute and store energies and angles
        KE_outer = kinetic_energy_outer(state.omega1)
        KE_inner = kinetic_energy_inner(state.omega2)
        PE_total = (
            potential_energy_outer(state.theta1)
            + potential_energy_coupling(state.theta1, state.theta2)
            + potential_energy_ring_ring(state.theta1, state.theta2)
        )
        KE_total = KE_outer + KE_inner
        total_energy = KE_total + PE_total
        state.energies.append((KE_outer, KE_inner, PE_total, KE_total, total_energy))
        state.angles.append((state.theta1, state.theta2))
        state.omegas.append((state.omega1, state.omega2))

        # Compute energy drift
        if state.initial_total_energy is None:
            state.initial_total_energy = total_energy
        energy_drift = total_energy - state.initial_total_energy
        state.drifts.append(energy_drift)

        # Append current time to times list
        state.times.append(state.time)

        # Determine indices for plotting based on max_history
        if max_history is not None:
            indices = slice(-max_history, None)
        else:
            indices = slice(None)

        # Update plots
        times = state.times[indices]
        KE_out, KE_in, PE_tot, KE_tot, E_tot = zip(*state.energies[indices])
        energy_lines["KE Outer"].set_data(times, KE_out)
        energy_lines["KE Inner"].set_data(times, KE_in)
        energy_lines["PE Total"].set_data(times, PE_tot)
        energy_lines["KE Total"].set_data(times, KE_tot)
        energy_lines["Total Energy"].set_data(times, E_tot)

        drift_line.set_data(times, state.drifts[indices])

        theta1_values, theta2_values = zip(*state.angles[indices])
        angle_lines["Theta1"].set_data(times, theta1_values)
        angle_lines["Theta2"].set_data(times, theta2_values)

        # Update axes limits dynamically
        if dynamic_lims:
            time_min, time_max = times[0], times[-1]
            ax_energy.set_xlim(time_min, time_max)
            ax_drift.set_xlim(time_min, time_max)
            ax_angles.set_xlim(time_min, time_max)

            current_max_drift = max(abs(min(state.drifts[indices], default=0)), max(state.drifts[indices], default=0)) * 1.1
            ax_drift.set_ylim(-current_max_drift, current_max_drift)

            current_max_energy = max(max(E_tot), 1) * 1.1
            ax_energy.set_ylim(0, current_max_energy)

            current_max_angle = max(abs(min(theta1_values + theta2_values, default=0)), max(theta1_values + theta2_values, default=0)) * 1.1
            ax_angles.set_ylim(-current_max_angle, current_max_angle)

        return line1, line2, *energy_lines.values(), drift_line, *angle_lines.values()

    # Start time for simulation
    start_time = time.time()

    # Create animation
    ani = FuncAnimation(
        fig, update, frames=frames, blit=True,
        interval=1000 / fps_video,  # Interval in milliseconds
        fargs=(state,)
    )

    # If save_filename is specified, save the animation
    if save_filename is not None:
        from matplotlib.animation import FFMpegWriter
        writer = FFMpegWriter(fps=fps_video)
        ani.save(save_filename, writer=writer)
        print(f"Animation saved to {save_filename}")
    else:
        # Show the animations
        plt.show()

    # Print simulation time
    end_time = time.time()
    print(f"Simulation run time: {end_time - start_time:.2f} seconds")
    if frame_limit is not None and frame_limit > 0:
        total_frames = frame_limit
        print(f"CPU Time per frame: {(end_time - start_time) / total_frames:.4f} seconds")

    # If data_filename is specified, save the data
    if data_filename is not None:
        import csv
        with open(data_filename, 'w', newline='') as csvfile:
            fieldnames = ['time', 'theta1', 'theta2', 'omega1', 'omega2',
                          'KE_outer', 'KE_inner', 'PE_total', 'KE_total', 'total_energy', 'energy_drift']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(len(state.energies)):
                row = {
                    'time': state.times[i],
                    'theta1': state.angles[i][0],
                    'theta2': state.angles[i][1],
                    'omega1': state.omegas[i][0],
                    'omega2': state.omegas[i][1],
                    'KE_outer': state.energies[i][0],
                    'KE_inner': state.energies[i][1],
                    'PE_total': state.energies[i][2],
                    'KE_total': state.energies[i][3],
                    'total_energy': state.energies[i][4],
                    'energy_drift': state.drifts[i],
                }
                writer.writerow(row)
        print(f"Data saved to {data_filename}")
        # Also save initial parameters
        param_filename = data_filename + '_params.txt'
        with open(param_filename, 'w') as param_file:
            param_file.write("Parameters (SI units):\n")
            param_file.write(f"k = {k} N·m/rad\n")
            param_file.write(f"U_rr = {U_rr} N·m\n")
            param_file.write(f"R1 = {R1} m, R2 = {R2} m\n")
            param_file.write(f"r = {r} m\n")
            param_file.write(f"Density = {rho} kg/m³\n")
            param_file.write(f"dt = {dt} s\n")
            param_file.write(f"Inner object: {inner_object_description}\n\n")
            param_file.write("Initial Conditions (SI units):\n")
            param_file.write(f"theta1_0 = {theta1_0} rad\n")
            param_file.write(f"omega1_0 = {omega1_0} rad/s\n")
            param_file.write(f"theta2_0 = {theta2_0} rad\n")
            param_file.write(f"omega2_0 = {omega2_0} rad/s\n")
        print(f"Parameters saved to {param_filename}")

    # At the end of the function, close the figure to prevent memory leaks
    plt.close(fig)

def run_single_simulation(params):
    i, sim_params, total_sims = params
    print(f"\nRunning Simulation {i}/{total_sims}")
    output_prefix = f"simulation_{i}"
    sim_params['save_filename'] = f"{output_prefix}.mp4"
    sim_params['data_filename'] = f"{output_prefix}_data.csv"
    sim_params['frame_limit'] = int(30 / sim_params['dt'])  # 30 seconds total
    sim_params['max_history'] = None  # Uncapped history
    sim_params['dynamic_lims'] = True
    simulate_coupled_rings(**sim_params)

def run_simulations(simulations):
    """
    Run multiple simulations in parallel.

    Parameters:
    simulations : list of dict
        List of dictionaries containing the parameters for each simulation.

    Returns:
    None
    """
    total_sims = len(simulations)
    sim_params_with_indices = [(i+1, sim_params, total_sims) for i, sim_params in enumerate(simulations)]

    # Use multiprocessing.Pool to run simulations in parallel
    with multiprocessing.Pool() as pool:
        pool.map(run_single_simulation, sim_params_with_indices)

if __name__ == '__main__':
    # Define simulation parameters
    simulations = []

    # Common parameters
    common_params = {
        'dt': 0.001,
        'R1': 0.12,     # Outer ring radius in meters (max 15 cm)
        'R2': 0.08,     # Inner ring/disk radius in meters
        'r': 0.005,     # Cross-sectional radius or half-thickness in meters
        'theta1_0': 0.0,  # Initial angle of outer ring
        'theta2_0': 0.0,  # Initial angle of inner ring/disk
        'k': 0.01,      # Torsional spring constant in N·m/rad
    }

    # Variation parameters
    U_rr_values = [0.1, 0.5, 1.0]  # Different U_rr values
    density_values = [7800.0]  # Density for steel
    is_disk_values = [False, True]  # Inner object is a ring or a disk
    initial_omegas = [0.0, 1.0, 5.0]  # Initial angular velocities (rad/s)

    # Generate simulations
    for U_rr in U_rr_values:
        for rho in density_values:
            for is_disk in is_disk_values:
                for omega_value in initial_omegas:
                    # Case 1: Vary omega1_0, omega2_0 = 0
                    sim_params = common_params.copy()
                    sim_params['U_rr'] = U_rr
                    sim_params['rho'] = rho
                    sim_params['is_disk'] = is_disk
                    sim_params['omega1_0'] = omega_value
                    sim_params['omega2_0'] = 0.0
                    simulations.append(sim_params)

                    # Case 2: Vary omega2_0, omega1_0 = 0
                    sim_params = common_params.copy()
                    sim_params['U_rr'] = U_rr
                    sim_params['rho'] = rho
                    sim_params['is_disk'] = is_disk
                    sim_params['omega1_0'] = 0.0
                    sim_params['omega2_0'] = omega_value
                    simulations.append(sim_params)

    # Run simulations in parallel
    run_simulations(simulations)