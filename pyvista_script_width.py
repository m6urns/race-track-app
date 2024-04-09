import numpy as np
import pyvista as pv

# Read the CSV data
data = np.genfromtxt('./tracks/Oschersleben_SCurve.csv', delimiter=',', skip_header=1)
# data = np.genfromtxt('./tracks/IMS_Straight.csv', delimiter=',', skip_header=1)
# data = np.genfromtxt('./tracks/Austin_Straight.csv', delimiter=',', skip_header=1)
# data = np.genfromtxt('./tracks/IMS_UTurn.csv', delimiter=',', skip_header=1)

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

# Iterate through each segment to create the track
for i in range(len(x) - 1):
    start_point = [x[i], y[i], 0]
    end_point = [x[i+1], y[i+1], 0]
    
    segment_accel = (acceleration_norm[i] + acceleration_norm[i+1]) / 2
    radius = base_radius + segment_accel * base_radius
    
    segment_points = np.array([start_point, end_point])
    line = pv.lines_from_points(segment_points)
    
    tube = line.tube(radius=radius)
    tube["acceleration"] = np.full(tube.n_points, segment_accel)
    
    multi_block.append(tube)

# Combine all tube segments into a single mesh
combined_mesh = multi_block.combine()

# Add the track mesh to the plotter
# plotter.add_mesh(combined_mesh, scalars='acceleration', cmap='plasma', label='Race Track')
# plotter.add_mesh(combined_mesh, scalars='acceleration', cmap='YlGnBu', label='Race Track')
plotter.add_mesh(combined_mesh, scalars='acceleration', cmap='magma', label='Race Track')
# plotter.add_mesh(combined_mesh, scalars='acceleration', cmap='inferno', label='Race Track')

# Color blind friendly color map
# plotter.add_mesh(combined_mesh, scalars='acceleration', cmap='cividis', label='Race Track')

# Calculate the direction vector for the arrow
direction = np.array([x[1] - x[0], y[1] - y[0], 0])
direction_norm = direction / np.linalg.norm(direction)

# Calculate a perpendicular vector to offset the arrow's start position
perpendicular = np.array([-direction_norm[1], direction_norm[0], 0])
offset_distance = 50  # Adjust as needed
arrow_start = np.array([x[0], y[0], 0]) + perpendicular * offset_distance

# Create and add the arrow to the plotter
arrow = pv.Arrow(start=arrow_start, direction=direction_norm, scale=75)  # Adjust scale as needed
plotter.add_mesh(arrow, color='red')

# Set background color and show the plot
plotter.set_background('grey')
plotter.show()
