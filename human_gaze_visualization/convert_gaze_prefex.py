from PIL import Image
import os
from tqdm import tqdm
import shutil
# 将图片后缀从.jpg更改为.png
def change_image_suffix(input_dir, output_dir):
    for filename in tqdm(os.listdir(input_dir)):
        if filename.startswith("first_aoi_"):
            shutil.copyfile(os.path.join(input_dir, filename), os.path.join(input_dir, filename.replace("first_aoi_", "")))
            os.remove(os.path.join(input_dir, filename))

# 使用示例
input_dir = 'all_new_old_driver_other/first_aoi_end/old'
output_dir = 'all_new_old_driver_other/first_aoi/new'
os.makedirs(output_dir, exist_ok=True)
change_image_suffix(input_dir, output_dir)