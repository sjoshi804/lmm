from PIL import Image
from datasets import load_from_disk
import random 

dataset = load_from_disk('/data/lmm/generated/v3_spatial_grid_multimodal')

#Parse grid
def parse_grid(grid_str, K):
    """
    Parse the grid string into a 2D list of grid cells.
    """
    grid_str = '\n'.join(grid_str.split('\n')[:K])
    rows = grid_str.strip().split('\n')
    return [[cell.strip() for cell in row.split('|') if cell.strip()] for row in rows]

BORDER_SIZE = 6

def merge_image(grid, final_size=(256, 256), num_unique_images = 1000):
    # Determine the number of rows and columns
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    
    # Calculate available width and height for each image
    total_border_width = (cols - 1) * BORDER_SIZE
    total_border_height = (rows - 1) * BORDER_SIZE
    available_width = final_size[0] - total_border_width
    available_height = final_size[1] - total_border_height
    element_width = available_width // cols
    element_height = available_height // rows
    
    # Create the new image with a black background
    combined_image = Image.new('RGB', final_size, 'black')
    
    # Place each image in the combined image
    for row_index, row in enumerate(grid):
        for col_index, word in enumerate(row):
            i = random.randint(0, num_unique_images)
            img = Image.open(f"/home/allanz/data/images/{word}/{i}.png").resize((element_height, element_width), Image.Resampling.LANCZOS)
            x = col_index * (element_width + BORDER_SIZE)
            y = row_index * (element_height + BORDER_SIZE)
            combined_image.paste(img, (x, y))
    
    return combined_image

def save_merged_grid(set_type, num_unique_images = 1000):
    grids = dataset[set_type]["text"]
    count = 0 
    for grid in grids:
        temp_grid = parse_grid(grid, 3)
        img = merge_image(temp_grid, num_unique_images = num_unique_images)
        img.save(f"/home/allanz/data/grid/{set_type}/{count}.png")
        count += 1
        print(f"Grid {count} for {set_type} saved successfully!")
    print(f"Grids for {set_type} saved successfully!")

save_merged_grid("train")
save_merged_grid("test")
save_merged_grid("validation")


