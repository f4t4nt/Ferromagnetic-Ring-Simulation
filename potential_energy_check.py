import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# Set matplotlib style
plt.style.use('ggplot')

# Constants
R1 = 1  # Radius of the outer ring (Ring 1)
R2_values = [0.999, 0.99, 0.9, 3/4, 2/3, 1/2, 1/3, 1/4, 0.001]  # List of radii for the inner ring (Ring 2)
theta_values_quarter = np.linspace(0, np.pi/2, 500)  # Compute only from 0 to pi/2
csv_filename = "potential_energy_data.csv"

# Number of points for alpha and beta in the numerical integration
num_points = 1000
alpha = np.linspace(0, 2 * np.pi, num_points)       # Angular positions on Ring 1
beta = np.linspace(0, 2 * np.pi, num_points)        # Angular positions on Ring 2
alpha_grid, beta_grid = np.meshgrid(alpha, beta)    # Create a grid for alpha and beta

# Prepare figure
fig = plt.figure(figsize=(14, 10))
gs_main = fig.add_gridspec(2, 1, height_ratios=[1, 4])
fig.suptitle('Potential Energy Integral Analysis', fontsize=16)

# Parameters axis
ax_params = fig.add_subplot(gs_main[0])
ax_params.axis('off')
param_text = ax_params.text(
    0, 1,
    (
        f"Parameters:\n"
        f"$R_1 = {R1}$\n"
        f"Computed $R_2$ values: {', '.join(f'{R2:.2f}' for R2 in R2_values)}\n"
        f"Number of points in integration: {num_points}\n"
        f"Theta range: $0$ to $2\\pi$\n\n"
        "Each integral plot is normalized to have min = 0 and max = 1."
    ),
    fontsize=12, verticalalignment='top', transform=ax_params.transAxes
)

# Potential Energy Plot axis
ax_pe = fig.add_subplot(gs_main[1])

# Initialize a dictionary to store original min and max values
original_min_max = {}

# Initialize or load results DataFrame
if os.path.exists(csv_filename):
    # Load precomputed data from CSV
    data = pd.read_csv(csv_filename)
else:
    # Initialize a new DataFrame
    data = pd.DataFrame({'delta_theta_quarter': theta_values_quarter})

# Loop over different values of R2 (inner ring radii)
for R2 in R2_values:
    column_name = f"R2_{R2}"
    if column_name in data.columns:
        # Use the precomputed data
        I_theta_quarter = data[column_name].values
        # Reflect and translate data to generate the full range
        I_theta_half = np.concatenate([I_theta_quarter, I_theta_quarter[::-1]])
        I_theta_full = np.concatenate([I_theta_half, I_theta_half])
        theta_values_full = np.linspace(0, 2 * np.pi, len(I_theta_full))
        # Store original min and max
        original_min = np.min(I_theta_full)
        original_max = np.max(I_theta_full)
        original_min_max[R2] = (original_min, original_max)
        # Normalize the data
        I_theta_normalized = (I_theta_full - original_min) / (original_max - original_min)
        # Plot the normalized potential energy integral
        ax_pe.plot(theta_values_full, I_theta_normalized, label=f'$R_2 = {R2:.3f}$')
    else:
        # Compute data for this R2 value
        I_theta_quarter = []  # List to store the potential energy integral for each theta (0 to pi/2)
        # Loop over theta values (relative orientation angle between the rings)
        for theta in theta_values_quarter:
            # Precompute trigonometric functions for efficiency
            sin_alpha = np.sin(alpha_grid)      # sin(α) grid
            cos_alpha = np.cos(alpha_grid)      # cos(α) grid
            sin_beta = np.sin(beta_grid)        # sin(β) grid
            cos_beta = np.cos(beta_grid)        # cos(β) grid
            cos_theta = np.cos(theta)           # cos(θ)
            sin_theta = np.sin(theta)           # sin(θ)

            # Compute the differences in x, y, z components between differential charge elements
            x_diff = R1 * sin_alpha - R2 * sin_beta * cos_theta
            y_diff = R1 * cos_alpha - R2 * cos_beta
            z_diff = - R2 * sin_beta * sin_theta  # Note: z-coordinate of Ring 1 is zero

            # Compute the distance between charge elements (r_12)
            r = np.sqrt(x_diff**2 + y_diff**2 + z_diff**2)

            # Avoid division by zero by setting the integrand to zero where r is zero
            with np.errstate(divide='ignore', invalid='ignore'):
                integrand = np.where(r != 0, 1 / r, 0)

            # Perform the double integral numerically using the composite trapezoidal rule
            I = np.trapz(
                    np.trapz(integrand, beta, axis=0), alpha)

            # Multiply by R1 * R2 (since we've dropped the constants k and λ^2 for clarity)
            I_total = R1 * R2 * I

            # Append the total integral value for this theta to the list
            I_theta_quarter.append(I_total)

        # Add this R2's data to DataFrame
        data[column_name] = I_theta_quarter

        # Reflect and translate data to generate the full range
        I_theta_half = np.concatenate([I_theta_quarter, I_theta_quarter[::-1]])
        I_theta_full = np.concatenate([I_theta_half, I_theta_half])
        theta_values_full = np.linspace(0, 2 * np.pi, len(I_theta_full))

        # Store original min and max
        original_min = np.min(I_theta_full)
        original_max = np.max(I_theta_full)
        original_min_max[R2] = (original_min, original_max)

        # Normalize the data
        I_theta_normalized = (I_theta_full - original_min) / (original_max - original_min)

        # Plot the normalized potential energy integral
        ax_pe.plot(theta_values_full, I_theta_normalized, label=f'$R_2 = {R2:.3f}$')

# Save updated data to CSV
data.to_csv(csv_filename, index=False)

# Plot cos(2Δθ) for comparison, scaled to match the data (no normalization)
theta_values_cos = np.linspace(0, 2 * np.pi, 1000)
cos_2theta = (1 + np.cos(2 * theta_values_cos)) / 2 
ax_pe.plot(theta_values_cos, cos_2theta, 'k:', linewidth=2, label='$\\cos(2\\Delta\\theta)$')

# Plot (1 - |sin(Δθ)|)^3 for comparison
sin_pow = 2
sin_x = np.sin(theta_values_cos)
sin_x_abs = np.abs(sin_x)
one_minus_sin_x_abs_pow = (1 - sin_x_abs)**sin_pow
ax_pe.plot(theta_values_cos, one_minus_sin_x_abs_pow, 'b:', linewidth=2, label=f'$(1-|\\sin(\\Delta\\theta)|)^{sin_pow}$')

# Configure the plot with title, labels, legend, and grid
ax_pe.set_title('Normalized Potential Energy Integral $U(\\Delta \\theta)$ vs Relative Orientation $\\Delta \\theta = \\theta_1 - \\theta_2$', fontsize=14)
ax_pe.set_xlabel('$\\Delta \\theta$ (radians)', fontsize=12)
ax_pe.set_ylabel('Normalized Potential Energy $U(\\Delta \\theta)$', fontsize=12)
ax_pe.grid(True)

# Place legend
ax_pe.legend(loc='upper right', fontsize=10)

# Adjust layout
fig.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save the plot
plt.savefig('potential_energy_plot.png', dpi=300)