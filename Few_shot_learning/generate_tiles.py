from PIL import Image
from tqdm import tqdm

filename = "input/River.jpg"

tile_size = 256

# load file as img
with Image.open(filename) as img:
    img.load()

# get image dimensions
image_width, image_height = img.size

# determine number of tiles in image
num_rows = image_height // tile_size
num_cols = image_width // tile_size

for i in tqdm(range(num_rows),desc="Generating Tiles"):
    for j in range(num_cols):
        # crop parameters define left, upper, right, and bottom edges of tile
        cropped = img.crop((j*tile_size,i*tile_size,j*tile_size+tile_size,i*tile_size+tile_size))
        name = "output/River_" + str(i) + "_" + str(j) + ".png"
        cropped.save(name)