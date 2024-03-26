from PIL import Image
import os
from tqdm import tqdm

# 将图片后缀从.jpg更改为.png
def change_image_suffix(input_dir, output_dir):
    for filename in tqdm(os.listdir(input_dir)):
        with Image.open(os.path.join(input_dir, filename)) as im:
            new_filename = os.path.splitext(filename)[0] + '.png'
            im.save(os.path.join(output_dir, new_filename), 'png')

# 使用示例
input_dir = 'human_machine_gaze_data/all_raw_image_v3'
output_dir = 'human_machine_gaze_data/all_raw_image_v3_png'
os.makedirs(output_dir, exist_ok=True)
change_image_suffix(input_dir, output_dir)