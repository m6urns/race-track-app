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

# Create sphere glyphs with sizes based on acceleration magnitude
sphere_size_factor = 0.01
sphere_sizes = acceleration * sphere_size_factor
spheres = pv.PolyData(points)
spheres.point_data['acceleration'] = acceleration
spheres.point_data['sphere_sizes'] = sphere_sizes  # Assign sphere sizes to point data
glyphs = spheres.glyph(geom=pv.Sphere(), scale='sphere_sizes')  # Reference the scale by the name of the new data array
# Add sphere glyphs to the plotter
plotter.add_mesh(glyphs, scalars='acceleration', cmap='coolwarm', label='Acceleration')

# Add a scalar bar for acceleration values
# plotter.add_scalar_bar(title='Acceleration', n_labels=5, position_x=0.85, position_y=0.05, width=0.1, height=0.5)

# Set background color and show the plot
plotter.set_background('white')
plotter.show()