
from PIL import Image

def is_image_file(filename):
    """判断文件是否为支持的图像格式（含 .tif，用于 CSD 数据集）"""
    return any(filename.endswith(ext) for ext in
               [".png", ".jpg", ".bmp", ".JPG", ".jpeg", ".tif", ".tiff", ".TIF", ".TIFF"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img
