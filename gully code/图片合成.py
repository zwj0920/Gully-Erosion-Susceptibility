from PIL import Image
import matplotlib.pyplot as plt

def merge_images_with_gap(image_paths, output_path, dpi=600, grid_size=(2, 3), gap=5, preview=False):
    """
    合并多张图片并保存输出，支持设置DPI和图片间隙，可预览排版。

    参数：
    image_paths (list of str): 要合并的图片文件路径列表。
    output_path (str): 输出图片保存路径。
    dpi (int, optional): 输出图片的DPI，默认值为600。
    grid_size (tuple, optional): 输出图片的网格尺寸，默认值为3行2列。
    gap (int, optional): 每张图片之间的间隙，默认值为5像素。
    preview (bool, optional): 是否预览排版，默认值为False。
    """
    # 打开图片并存储到列表中
    images = [Image.open(image_path) for image_path in image_paths]
    
    # 计算每张图片的宽度和高度（假设所有图片尺寸相同）
    img_width, img_height = images[0].size
    
    # 计算合并后的图片总宽度和总高度，包含间隙
    grid_cols, grid_rows = grid_size
    total_width = img_width * grid_cols + gap * (grid_cols - 1)
    total_height = img_height * grid_rows + gap * (grid_rows - 1) // 16

    # 创建一个新的空白图片，尺寸为计算所得的总宽度和总高度
    merged_image = Image.new('RGB', (total_width, total_height), (255, 255, 255))  # 设置白色背景
    
    # 将每张图片依次粘贴到新图片中
    for idx, img in enumerate(images):
        row = idx // grid_cols
        col = idx % grid_cols
        x_offset = col * (img_width + gap)
        y_offset = row * (img_height + gap // 16)
        merged_image.paste(img, (x_offset, y_offset))

    # 预览排版
    if preview:
        plt.figure(figsize=(15, 12))  # 增加预览窗口的大小
        plt.imshow(merged_image)
        plt.axis('off')
        plt.title("合并图片预览（带间隙）", fontsize=20)  # 增大标题字体
        plt.show()

    # 保存合并后的图片并设置指定的DPI
    merged_image.save(output_path, dpi=(dpi, dpi))


# 示例用法
image_paths = [
    r"C:\Users\huang\Desktop\论文图片\glmm模型\拟合图\沟宽.png",
    r"C:\Users\huang\Desktop\论文图片\glmm模型\拟合图\沟深.png",
    r"C:\Users\huang\Desktop\论文图片\glmm模型\拟合图\沟长.png",
    r"C:\Users\huang\Desktop\论文图片\glmm模型\拟合图\周长.png",
    r"C:\Users\huang\Desktop\论文图片\glmm模型\拟合图\面积.png",
    r"C:\Users\huang\Desktop\论文图片\glmm模型\拟合图\体积.png"
]  # 替换为你的图片路径
output_path = r"C:\Users\huang\Desktop\论文图片\glmm模型\拟合图\总图.png"  # 输出图片路径
merge_images_with_gap(image_paths, output_path, dpi=600, preview=True)
