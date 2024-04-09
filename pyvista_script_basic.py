import numpy as np
import pyvista as pv

# Read the CSV data
data = np.genfromtxt('./tracks/Oschersleben_SCurve.csv', delimiter=',', skip_header=1)
x = data[:, 0]
y = data[:, 1]
acceleration = data[:, 2]

# Create a plotter
plotter = pv.Plotter()

# Create a line representing the race track
points = np.column_stack((x, y, np.zeros_like(x)))  # Assuming 2D data, set z-coordinate to 0
poly = pv.PolyData()
poly.points = points
cells = np.full((len(points)-1, 3), 2, dtype=np.int_)
cells[:, 1] = np.arange(0, len(points)-1, dtype=np.int_)
cells[:, 2] = np.arange(1, len(points), dtype=np.int_)
poly.lines = cells
plotter.add_mesh(poly, color='black', line_width=5, label='Race Track')

# Create a scatter plot of acceleration datapoints
acceleration_points = pv.PolyData(points)
acceleration_points.point_data['acceleration'] = acceleration
plotter.add_mesh(acceleration_points, scalars='acceleration', cmap='coolwarm', point_size=10, label='Acceleration')

# Add a scalar bar for acceleration values
# plotter.add_scalar_bar(title='Acceleration', n_labels=5, position_x=0.85, position_y=0.05, width=0.1, height=0.5)

# Set background color and show the plot
plotter.set_background('white')
plotter.show()