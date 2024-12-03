from collections import deque
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import time

def simulate_coupled_rings(
    k=1.0,                      # Torsional spring constant (N·m/rad)
    U_rr=1.0,                   # Magnetic interaction potential coefficient (N·m)
    rho=1.0,                    # Density of the material (kg/m³)
    R1=1.5,                     # Radius of the outer ring (m)
    R2=1.0,                     # Radius of the inner ring or disk (m)
    r=0.1,                      # Cross-sectional radius of the rings or half-thickness of the disk (m)
    is_disk=False,              # If True, inner object is a disk; if False, a ring
    theta1_0=0.0,               # Initial angle of the outer ring (rad)
    omega1_0=10.0,              # Initial angular velocity of the outer ring (rad/s)
    theta2_0=0.0,               # Initial angle of the inner ring or disk (rad)
    omega2_0=0.0,               # Initial angular velocity of the inner ring or disk (rad/s)
    dt=0.01,                    # Time step for the simulation (s)
    max_history_energy=10,      # Maximum history time to store for energy plot (seconds)
    max_history_angles=10,      # Maximum history time to store for angles plot (seconds)
    max_history_drift=10,       # Maximum history time to store for drift plot (seconds)
    save_filename=None,         # Filename to save the animation
    data_filename=None,         # Filename to save the data
    frame_limit=6000,           # Number of frames to simulate
    dynamic_lims=True,          # Whether to adjust axes limits dynamically
    fps_video=60,               # Frames per second for the saved video
):
    # Check for valid parameters
    if (save_filename is not None or data_filename is not None) and frame_limit is None:
        raise ValueError("frame_limit must be specified if save_filename or data_filename is specified.")

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

            # Initialize separate deques for each plot, with appropriate maxlen
            self.times_energy = deque(maxlen=int(max_history_energy / dt))
            self.energies = deque(maxlen=int(max_history_energy / dt))

            self.times_angles = deque(maxlen=int(max_history_angles / dt))
            self.angles = deque(maxlen=int(max_history_angles / dt))
            self.omegas = deque(maxlen=int(max_history_angles / dt))

            self.times_drift = deque(maxlen=int(max_history_drift / dt))
            self.drifts = deque(maxlen=int(max_history_drift / dt))

            self.time = 0.0
            self.initial_total_energy = None

            # For saving data
            self.all_times = []
            self.all_energies = []
            self.all_angles = []
            self.all_omegas = []
            self.all_drifts = []

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
        return U_rr * np.cos(2 * (theta1 - theta2)) + U_rr

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
    ax_energy.set_ylim(0, 0.1)
    ax_drift.set_ylim(-0.1, 0.1)
    ax_angles.set_ylim(-np.pi, np.pi)

    # Adjust layout
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Simulation Parameters
    N_steps_per_frame = max(1, int(1 / (dt * fps_video)))  # Number of simulation steps per frame
    print(f"N_steps_per_frame: {N_steps_per_frame}")

    if frame_limit is not None:
        total_frames = frame_limit // N_steps_per_frame
        frames = range(total_frames)
    else:
        frames = None  # For indefinite simulation

    # Initialize current max values for dynamic limits
    current_max_energy = 0
    current_max_drift = 0.1
    current_max_angle = np.pi

    # Animation update function
    def update(frame, state):
        nonlocal current_max_energy, current_max_drift, current_max_angle

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

            # Compute energies and angles
            KE_outer = kinetic_energy_outer(state.omega1)
            KE_inner = kinetic_energy_inner(state.omega2)
            PE_total = (
                potential_energy_outer(state.theta1)
                + potential_energy_coupling(state.theta1, state.theta2)
                + potential_energy_ring_ring(state.theta1, state.theta2)
            )
            KE_total = KE_outer + KE_inner
            total_energy = KE_total + PE_total

            # Compute energy drift
            if state.initial_total_energy is None:
                state.initial_total_energy = total_energy
            energy_drift = total_energy - state.initial_total_energy

            # Append data to deques
            # Energy data
            state.times_energy.append(state.time)
            state.energies.append((KE_outer, KE_inner, PE_total, KE_total, total_energy))

            # Angles data
            state.times_angles.append(state.time)
            state.angles.append((state.theta1, state.theta2))
            state.omegas.append((state.omega1, state.omega2))

            # Drift data
            state.times_drift.append(state.time)
            state.drifts.append(energy_drift)

            # Save all data for data export
            state.all_times.append(state.time)
            state.all_energies.append((KE_outer, KE_inner, PE_total, KE_total, total_energy))
            state.all_angles.append((state.theta1, state.theta2))
            state.all_omegas.append((state.omega1, state.omega2))
            state.all_drifts.append(energy_drift)

        # Update line positions
        x2 = [-line_length2 * np.cos(state.theta2), line_length2 * np.cos(state.theta2)]
        y2 = [-line_length2 * np.sin(state.theta2), line_length2 * np.sin(state.theta2)]
        line2.set_data(x2, y2)
        x1 = [-line_length1 * np.cos(state.theta1), line_length1 * np.cos(state.theta1)]
        y1 = [-line_length1 * np.sin(state.theta1), line_length1 * np.sin(state.theta1)]
        line1.set_data(x1, y1)

        # Extract data for energy plot
        times_energy = list(state.times_energy)
        KE_out_values = [e[0] for e in state.energies]
        KE_in_values = [e[1] for e in state.energies]
        PE_tot_values = [e[2] for e in state.energies]
        KE_tot_values = [e[3] for e in state.energies]
        E_tot_values = [e[4] for e in state.energies]

        # Extract data for angles plot
        times_angles = list(state.times_angles)
        theta1_values = [a[0] for a in state.angles]
        theta2_values = [a[1] for a in state.angles]

        # Extract data for drift plot
        times_drift = list(state.times_drift)
        drift_values = list(state.drifts)

        # Update plots
        energy_lines["KE Outer"].set_data(times_energy, KE_out_values)
        energy_lines["KE Inner"].set_data(times_energy, KE_in_values)
        energy_lines["PE Total"].set_data(times_energy, PE_tot_values)
        energy_lines["KE Total"].set_data(times_energy, KE_tot_values)
        energy_lines["Total Energy"].set_data(times_energy, E_tot_values)

        angle_lines["Theta1"].set_data(times_angles, theta1_values)
        angle_lines["Theta2"].set_data(times_angles, theta2_values)

        drift_line.set_data(times_drift, drift_values)

        # Update axes limits dynamically
        if dynamic_lims:
            if times_energy:
                time_min_energy = times_energy[0]
                time_max_energy = times_energy[-1]
                ax_energy.set_xlim(time_min_energy, time_max_energy)

                new_max_energy = max(E_tot_values) * 1.1
                if new_max_energy > current_max_energy:
                    current_max_energy = new_max_energy
                ax_energy.set_ylim(0, current_max_energy)

            if times_angles:
                time_min_angles = times_angles[0]
                time_max_angles = times_angles[-1]
                ax_angles.set_xlim(time_min_angles, time_max_angles)

                combined_angles = theta1_values + theta2_values
                new_max_angle = max(abs(min(combined_angles, default=0)), max(combined_angles, default=0)) * 1.1
                if new_max_angle > current_max_angle:
                    current_max_angle = new_max_angle
                ax_angles.set_ylim(-current_max_angle, current_max_angle)

            if times_drift:
                time_min_drift = times_drift[0]
                time_max_drift = times_drift[-1]
                ax_drift.set_xlim(time_min_drift, time_max_drift)

                new_max_drift = max(max(abs(min(drift_values, default=0)), max(drift_values, default=0)) * 1.1, new_max_energy * 0.1)
                if new_max_drift > current_max_drift:
                    current_max_drift = new_max_drift
                ax_drift.set_ylim(-current_max_drift, current_max_drift)

        return [line1, line2] + list(energy_lines.values()) + [drift_line] + list(angle_lines.values())

    # Start time for simulation
    start_time = time.time()

    # Set blit to False to ensure compatibility when saving
    blit = False

    # Create animation
    ani = FuncAnimation(
        fig, update, frames=frames, blit=blit,
        interval=1000 / fps_video,  # Interval in milliseconds
        fargs=(state,)
    )

    # If save_filename is specified, save the animation
    if save_filename is not None:
        from matplotlib.animation import FFMpegWriter
        writer = FFMpegWriter(fps=fps_video)
        dpi_value = 200  # Increase DPI for better quality

        # Add progress bar
        total_frames = len(frames)
        pbar = tqdm(total=total_frames, desc='Saving animation', unit='frame')

        def progress_callback(current_frame, total_frames):
            pbar.update(current_frame - pbar.n)

        ani.save(save_filename, writer=writer, dpi=dpi_value, progress_callback=progress_callback)
        pbar.close()
        print(f"Animation saved to {save_filename}")
    else:
        # Show the animation
        plt.show()

    # Print simulation time
    end_time = time.time()
    print(f"Simulation run time: {end_time - start_time:.2f} seconds")
    if frame_limit is not None and frame_limit > 0:
        total_frames = frame_limit // N_steps_per_frame
        print(f"CPU Time per frame: {(end_time - start_time) / total_frames:.4f} seconds")

    # If data_filename is specified, save the data
    if data_filename is not None:
        import csv
        with open(data_filename, 'w', newline='') as csvfile:
            fieldnames = ['time', 'theta1', 'theta2', 'omega1', 'omega2',
                          'KE_outer', 'KE_inner', 'PE_total', 'KE_total', 'total_energy', 'energy_drift']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(len(state.all_times)):
                row = {
                    'time': state.all_times[i],
                    'theta1': state.all_angles[i][0],
                    'theta2': state.all_angles[i][1],
                    'omega1': state.all_omegas[i][0],
                    'omega2': state.all_omegas[i][1],
                    'KE_outer': state.all_energies[i][0],
                    'KE_inner': state.all_energies[i][1],
                    'PE_total': state.all_energies[i][2],
                    'KE_total': state.all_energies[i][3],
                    'total_energy': state.all_energies[i][4],
                    'energy_drift': state.all_drifts[i],
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

    # Close the figure to prevent memory leaks
    plt.close(fig)

def run_single_simulation(params):
    i, sim_params, total_sims = params
    print(f"\nRunning Simulation {i}/{total_sims}")
    output_prefix = f"k={sim_params['k']}, r={sim_params['r']}, U_rr={sim_params['U_rr']}, rho={sim_params['rho']}, is_disk={sim_params['is_disk']}, omega1_0={sim_params['omega1_0']}, omega2_0={sim_params['omega2_0']}"
    sim_params['save_filename'] = f"{output_prefix}.mp4"
    sim_params['data_filename'] = f"{output_prefix}_data.csv"

    # Added: Check if the simulation outputs already exist
    if os.path.exists(sim_params['data_filename']) and os.path.exists(sim_params['save_filename']):
        print(f"Simulation {i}/{total_sims} already computed. Skipping.")
        return

    sim_params['frame_limit'] = int(60 / sim_params['dt'])
    sim_params['max_history_energy'] = 20
    sim_params['max_history_angles'] = 20
    sim_params['max_history_drift'] = 60
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
        'R1': 0.12,         # Outer ring radius in meters (max 15 cm)
        'R2': 0.08,         # Inner ring/disk radius in meters
        'r': 0.005,         # Cross-sectional radius or half-thickness in meters
        'theta1_0': 0.0,    # Initial angle of outer ring
        'theta2_0': 0.0,    # Initial angle of inner ring/disk
        'k': 0.005,         # Torsional spring constant in N·m/rad
    }

    # Variation parameters
    U_rr_values = [0.01, 0.05, 0.1]     # Different U_rr values
    density_values = [7800.0]           # Density for steel
    is_disk_values = [False, True]      # Inner object is a ring or a disk
    initial_omegas = [0.1, 1.0, 5.0]    # Initial angular velocities (rad/s)

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

# if __name__ == '__main__':
#     simulation_params = {
#         'k': 0.005,                   # Torsional spring constant (N·m/rad)
#         'U_rr': 0.01,                 # Magnetic interaction potential coefficient (N·m)
#         'rho': 1.0,                   # Density of the material (kg/m³)
#         'R1': 1.5,                    # Radius of the outer ring (m)
#         'R2': 1.0,                    # Radius of the inner ring or disk (m)
#         'r': 0.1,                     # Cross-sectional radius of the rings or half-thickness of the disk (m)
#         'is_disk': False,             # If True, inner object is a disk; if False, a ring
#         'theta1_0': 0.0,              # Initial angle of the outer ring (rad)
#         'omega1_0': 10.0,             # Initial angular velocity of the outer ring (rad/s)
#         'theta2_0': 0.0,              # Initial angle of the inner ring or disk (rad)
#         'omega2_0': 0.0,              # Initial angular velocity of the inner ring or disk (rad/s)
#         'dt': 0.01,                   # Time step for the simulation (s)
#         'max_history_energy': 20,     # Maximum history time to store for energy plot (seconds)
#         'max_history_angles': 20,     # Maximum history time to store for angles plot (seconds)
#         'max_history_drift': 100,     # Maximum history time to store for drift plot (seconds)
#         'save_filename': None,        # Filename to save the animation
#         'data_filename': None,        # Filename to save the data
#         'frame_limit': 6000,          # Number of frames to simulate
#         'dynamic_lims': True,         # Whether to adjust axes limits dynamically
#         'fps_video': 60,              # Frames per second for the saved video
#     }
    
#     simulate_coupled_rings(**simulation_params)