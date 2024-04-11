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

# Normalize the acceleration values to a suitable range for cube sizes
normalized_acceleration = (acceleration - acceleration.min()) / (acceleration.max() - acceleration.min())
cube_size_factor = 10  # Adjust this factor to scale the overall size of the cubes
scaled_cube_sizes = normalized_acceleration * cube_size_factor + 0.01  # Ensure there's a minimum size to avoid too small cubes

# Create cube glyphs with orientation based on acceleration direction
acceleration_vectors = np.column_stack((np.zeros_like(acceleration), np.zeros_like(acceleration), acceleration))
cubes = pv.PolyData(points)
cubes.point_data['acceleration'] = acceleration
cubes.point_data['scaled_cube_sizes'] = scaled_cube_sizes  # Assign scaled cube sizes to point data
cubes.point_data['orientation_vectors'] = acceleration_vectors  # Assign orientation vectors to point data
glyphs = cubes.glyph(geom=pv.Cube(), orient='orientation_vectors', scale='scaled_cube_sizes')  # Use scaled sizes

# Add cube glyphs to the plotter
plotter.add_mesh(glyphs, scalars='acceleration', cmap='coolwarm', label='Acceleration')

# Add a scalar bar for acceleration values
# plotter.add_scalar_bar(title='Acceleration', n_labels=5, position_x=0.85, position_y=0.05, width=0.1, height=0.5)

# Set background color and show the plot
plotter.set_background('white')
plotter.show()