import numpy as np
import pyvista as pv

# Read the CSV data
data = np.genfromtxt('./race-track-app/tracks/Oschersleben_SCurve.csv', delimiter=',', skip_header=1)
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

# Create cube glyphs with orientation based on acceleration direction
cube_size = 0.05
acceleration_vectors = np.column_stack((np.zeros_like(acceleration), np.zeros_like(acceleration), acceleration))
cubes = pv.PolyData(points)
cubes.point_data['acceleration'] = acceleration
glyphs = cubes.glyph(geom=pv.Cube(), orient=acceleration_vectors, scale=cube_size)

# Add cube glyphs to the plotter
plotter.add_mesh(glyphs, scalars='acceleration', cmap='coolwarm', label='Acceleration')

# Add a scalar bar for acceleration values
# plotter.add_scalar_bar(title='Acceleration', n_labels=5, position_x=0.85, position_y=0.05, width=0.1, height=0.5)

# Set background color and show the plot
plotter.set_background('white')
plotter.show()