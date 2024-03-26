import os
from PIL import Image, ImageDraw, ImageFont

folder_path = "."

image_width, image_height = 800, 600
num_rows, num_cols = 10, 10

collage_width = num_cols * image_width
collage_height = num_rows * image_height
collage = Image.new("RGB", (collage_width, collage_height), color="white")

font = ImageFont.load_default()

draw = ImageDraw.Draw(collage)

# Iterate over the PNG files in the folder
for index, filename in enumerate(os.listdir(folder_path)):
    if filename.endswith(".png"):
        image_path = os.path.join(folder_path, filename)
        image = Image.open(image_path)

        # Calculate the position of the image in the collage
        row = index // num_cols
        col = index % num_cols
        x = col * image_width
        y = row * image_height

        # Paste the image onto the collage
        collage.paste(image, (x, y))

        # Calculate the position of the filename text
        text_width, text_height = draw.textsize(filename, font=font)
        text_x = x + (image_width - text_width) // 2
        text_y = y + image_height - text_height - 10

        # Draw the filename text on the image
        draw.text((text_x, text_y), filename, font=font, fill="black")

# Save the collage image
collage.save("collage_accel.png")

