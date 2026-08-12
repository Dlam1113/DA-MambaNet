
from PIL import Image


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def is_image_file(filename):
    """不区分扩展名大小写地判断文件是否为项目支持的图像格式。"""
    return str(filename).lower().endswith(IMAGE_EXTENSIONS)


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img
