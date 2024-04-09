import numpy as np
import pyvista as pv

# Read the CSV data
data = np.genfromtxt('./tracks/Oschersleben_SCurve.csv', delimiter=',', skip_header=1)
x = data[:, 0]
y = data[:, 1]
acceleration = data[:, 2]

# Calculate tangent vectors along the track
tangents = np.gradient(np.column_stack((x, y)), axis=0)
tangents = tangents / np.linalg.norm(tangents, axis=1)[:, np.newaxis]

# Define the threshold angle (in degrees)
threshold_angle = 12
# threshold_angle = 170

# Define the rotation angle for the arrows (in degrees)
rotation_angle = 145

# Define the maximum acceleration threshold
max_acceleration_threshold = 198

# Create a plotter
plotter = pv.Plotter()

# Create a line representing the race track
points = np.column_stack((x, y, np.zeros_like(x)))
poly = pv.PolyData()
poly.points = points
cells = np.full((len(points)-1, 3), 2, dtype=np.int_)
cells[:, 1] = np.arange(0, len(points)-1, dtype=np.int_)
cells[:, 2] = np.arange(1, len(points), dtype=np.int_)
poly.lines = cells

# Add acceleration color map to the race track
poly.point_data['acceleration'] = acceleration
plotter.add_mesh(poly, scalars='acceleration', cmap='coolwarm', line_width=5, label='Race Track')

# Calculate the angle between acceleration vectors and track tangents
acceleration_vectors = np.column_stack((acceleration, np.zeros_like(acceleration), np.zeros_like(acceleration)))
dot_product = np.sum(acceleration_vectors[:, :2] * tangents, axis=1)
angles = np.degrees(np.arccos(dot_product / (np.linalg.norm(acceleration_vectors[:, :2], axis=1) * np.linalg.norm(tangents, axis=1))))

# Filter acceleration vectors based on the threshold angle and maximum acceleration
mask_angle = angles > threshold_angle
mask_acceleration = np.abs(acceleration) < max_acceleration_threshold
mask = mask_angle & mask_acceleration
filtered_points = points[mask]
filtered_acceleration = acceleration[mask]
filtered_acceleration_vectors = acceleration_vectors[mask]

# Check if there are any remaining acceleration vectors after filtering
if len(filtered_acceleration_vectors) > 0:
    # Rotate the acceleration vectors by the specified angle
    rotation_rad = np.radians(rotation_angle)
    rotation_matrix = np.array([[np.cos(rotation_rad), -np.sin(rotation_rad)],
                                [np.sin(rotation_rad), np.cos(rotation_rad)]])
    rotated_acceleration_vectors = np.dot(filtered_acceleration_vectors[:, :2], rotation_matrix)
    rotated_acceleration_vectors = np.column_stack((rotated_acceleration_vectors, np.zeros_like(rotated_acceleration_vectors[:, 0])))

    # Create arrow glyphs for acceleration
    arrow_size = 20
    arrows = pv.PolyData(filtered_points)
    arrows.point_data['acceleration'] = np.abs(filtered_acceleration)
    arrows.point_data['acceleration_vectors'] = rotated_acceleration_vectors

    # Normalize acceleration magnitudes for scaling
    acceleration_norm = np.linalg.norm(rotated_acceleration_vectors[:, :2], axis=1)
    if len(acceleration_norm) > 1:
        acceleration_norm = (acceleration_norm - acceleration_norm.min()) / (acceleration_norm.max() - acceleration_norm.min())
    else:
        acceleration_norm = np.ones_like(acceleration_norm)
    arrows.point_data['acceleration_norm'] = acceleration_norm

    # Set the 'acceleration_vectors' array as the active vectors
    arrows.set_active_vectors('acceleration_vectors')

    arrows_glyphs = arrows.glyph(geom=pv.Arrow(), orient=True, scale='acceleration_norm', factor=arrow_size)

    # Add arrow glyphs to the plotter with acceleration color map
    plotter.add_mesh(arrows_glyphs, scalars='acceleration', cmap='coolwarm', label='Acceleration')

    # Add a scalar bar for the acceleration color map
    plotter.add_scalar_bar(title='Acceleration', n_labels=5, position_x=0.85, position_y=0.05, width=0.1, height=0.5)

# Set background color and show the plot
plotter.set_background('white')
plotter.show()