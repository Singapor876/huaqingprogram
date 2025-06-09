import numpy as np
import rasterio
from PIL import Image

def compress_data(data, old_min=0, old_max=10000, new_min=0, new_max=255):
    return ((data - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min

tiff_path = r"D:\py\Tencent Files\2786593217\FileRecv\2019_1101_nofire_B2348_B12_10m_roi.tif"
try:
    with rasterio.open(tiff_path) as src:
        image_data = src.read()
        bands, height, width = image_data.shape
except Exception as e:
    print(f"无法加载文件: {e}")
    exit()

if bands != 5:
    raise ValueError("图片数据应该包含5个波段")

compressed_data = compress_data(image_data)

rgb_data = compressed_data[:3, :, :].astype(np.uint8)

rgb_data = np.transpose(rgb_data, (1, 2, 0))

rgb_image = Image.fromarray(rgb_data)
rgb_image.save("output_rgb_image.png")
##
