import numpy as np
import rasterio

def shuchu(tif_file):
    # 打开TIFF文件
    with rasterio.open(tif_file) as src:
        # 读取所有波段（假设波段顺序为B02, B03, B04, B08, B12）
        bands = src.read()  # 形状为 (波段数, 高度, 宽度)，这里是 (5, height, width)
        profile = src.profile  # 获取元数据

    # 分配波段（假设TIFF中的波段顺序为B02, B03, B04, B08, B12）
    blue = bands[0].astype(float)  # B02 - 蓝
    green = bands[1].astype(float)  # B03 - 绿
    red = bands[2].astype(float)  # B04 - 红
    nir = bands[3].astype(float)  # B08 - 近红外
    swir = bands[4].astype(float)  # B12 - 短波红外

    # 真彩色正则化
    rgb_orign = np.dstack((red, green, blue))

    # 确保数据范围在0-10000之间
    rgb_orign = np.clip(rgb_orign, 0, 10000)

    # 将数据范围从0-10000压缩到0-255
    rgb_normalized = (rgb_orign / 10000) * 255
    rgb_normalized = rgb_normalized.astype(np.uint8)

    return rgb_normalized, profile

# 调用函数
tif_file = '2019_1101_nofire_B2348_B12_10m_roi.tif'
rgb_image, profile = shuchu(tif_file)

# 修改元数据以保存为 RGB 图像
profile.update(
    dtype=rasterio.uint8,
    count=3,  # RGB 三个通道
    compress='lzw'  # 可选的压缩方式
)

# 保存图像
output_file = 'output_image.tif'
with rasterio.open(output_file, 'w', **profile) as dst:
    for i in range(3):
        dst.write(rgb_image[:, :, i], i + 1)

print(f"图像已保存到 {output_file}")