import numpy as np
import pyvista as pv

# Read the CSV data
data = np.genfromtxt('./tracks/Oschersleben_SCurve.csv', delimiter=',', skip_header=1)
x = data[:, 0]
y = data[:, 1]
acceleration = data[:, 2]

# Create a plotter
plotter = pv.Plotter()

# Normalize acceleration magnitudes for scaling
acceleration_norm = np.abs(acceleration)
if len(acceleration_norm) > 1:
    acceleration_norm = (acceleration_norm - acceleration_norm.min()) / (acceleration_norm.max() - acceleration_norm.min())
else:
    acceleration_norm = np.ones_like(acceleration_norm)

# Set a base radius for the tube
base_radius = 5

# Create a MultiBlock to hold all tube segments
multi_block = pv.MultiBlock()

# Iterate through each segment
for i in range(len(x) - 1):
    # Define the start and end points of the segment
    start_point = [x[i], y[i], 0]
    end_point = [x[i+1], y[i+1], 0]
    
    # Calculate the average acceleration for the segment
    segment_accel = (acceleration_norm[i] + acceleration_norm[i+1]) / 2
    
    # Adjust the radius based on the average acceleration
    radius = base_radius + segment_accel * base_radius  # Modify this formula as needed
    
    # Create a line for the segment
    segment_points = np.array([start_point, end_point])
    line = pv.lines_from_points(segment_points)
    
    # Create a tube for the segment
    tube = line.tube(radius=radius)
    
    # Set the scalar values for the entire tube to the average acceleration
    tube["acceleration"] = np.full(tube.n_points, segment_accel)  # Use n_points for point data

    # Add the tube to the MultiBlock
    multi_block.append(tube)

# Combine all tube segments into a single mesh
combined_mesh = multi_block.combine()

# Add the combined mesh to the plotter, using the acceleration scalar for coloring
plotter.add_mesh(combined_mesh, scalars='acceleration', cmap='coolwarm', label='Race Track')

# Add a scalar bar for the acceleration color map
plotter.add_scalar_bar(title='Acceleration', n_labels=5, position_x=0.85, position_y=0.05, width=0.1, height=0.5)

# Set background color and show the plot
plotter.set_background('white')
plotter.show()
