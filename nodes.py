import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch.nn.functional as F
import os # 导入os模块以检查字体路径

# ======================================================================
# 1. 更新后的绘制函数
# ======================================================================

def draw_glyph_flexible(font, text, width, height, max_font_size=140, num_text=1):
    if width <= 0:
        width = 1

    # height = max(width // 4, height)
    height = min(width // 6, height // num_text)
    
    img = Image.new(mode='1', size=(width, height), color=0)

    if not text or not text.strip():
        return img
        
    draw = ImageDraw.Draw(img)

    # --- Adaptive calculation of font size ---
    # Initial guess size
    g_size = 50
    new_font = font.font_variant(size=g_size)

    # get size
    left, top, right, bottom = new_font.getbbox(text)
    text_width_initial = max(right - left, 1)  
    text_height_initial = max(bottom - top, 1)

    width_ratio = width * 0.9 / text_width_initial
    height_ratio = height * 0.9 / text_height_initial
    ratio = min(width_ratio, height_ratio)
    final_font_size = int(g_size * ratio)
    
    # Apply the upper limit of font size
    if width > 1280:
        max_font_size = 180

    if width > 2048:
        max_font_size = 280

    final_font_size = min(final_font_size, max_font_size)
    new_font = font.font_variant(size=max(final_font_size, 10))

    draw.text(
        (width / 2, height / 2), 
        text, 
        font=new_font, 
        fill='white', 
        anchor='mm'  # Middle-Middle anchor
    )
    return img



# ======================================================================
# 2. 升级后的预处理器节点
# ======================================================================

class GlyphInpaintPreprocessor:
    """
    【新增】一个强大的预处理器，使用“叠罗汉”模式处理单行和多行文本。
    1. 它将输入文本分割成行。
    2. 它为每一行文本生成一个独立的字形图。
    3. 它将这些字形图垂直堆叠。
    4. 它将最终的字形图块堆叠在原始图像和遮罩的顶部。
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "text": ("STRING", {"multiline": True, "default": "第一行文字\n第二行文字"}),
                "font_path": ("STRING", {"default": "custom_nodes/textflux_comfyui/Arial-Unicode-Regular.ttf"}), 
                "max_font_size": ("INT", {"default": 140, "min": 10, "max": 1024, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("stacked_image", "stacked_mask")
    FUNCTION = "preprocess"
    CATEGORY = "Text Glyphs"

    def preprocess(self, image, mask, text, font_path, max_font_size):
        # --- 1. 准备工作 ---
        _b, height, width, _c = image.shape
        
        # 将输入文本分割成一个行列表
        text_lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not text_lines:
             # 如果没有文本，返回一个微小的黑条+原图，以避免流程中断
            blank_top = torch.zeros((_b, 1, width, _c), dtype=image.dtype, device=image.device)
            blank_mask_top = torch.zeros((_b, 1, width), dtype=mask.dtype, device=mask.device)
            return (torch.cat((blank_top, image), dim=1), torch.cat((blank_mask_top, mask), dim=1))
            
        num_lines = len(text_lines)
        
        # 加载字体
        try:
            if not os.path.exists(font_path):
                print(f"警告: 字体路径不存在 {font_path}，将使用默认字体。")
                raise IOError
            font = ImageFont.truetype(font_path, size=50)
        except IOError:
            font = ImageFont.load_default()
        
        # --- 2. 【新增】为每一行生成并收集字形图 ---
        glyph_tensors = []
        for line_text in text_lines:
            # 为每一行调用更新后的绘制函数
            glyph_pil = draw_glyph_flexible(
                font=font, 
                text=line_text, 
                width=width, 
                height=height, # 传入原图高度
                max_font_size=max_font_size,
                num_text=num_lines # 传入总行数
            )
            
            glyph_pil_rgb = glyph_pil.convert("RGB")
            glyph_np = np.array(glyph_pil_rgb).astype(np.float32) / 255.0
            glyph_tensor = torch.from_numpy(glyph_np).unsqueeze(0)
            glyph_tensors.append(glyph_tensor)
            
        # --- 3. 【新增】堆叠所有的字形图和遮罩 ---
        # 将所有独立的字形张量垂直堆叠（在高度维度上，即dim=1）
        stacked_glyphs = torch.cat(glyph_tensors, dim=1)
        
        # 创建一个与所有堆叠后的字形图总高度相同的纯黑遮罩
        _glyph_b, total_glyph_h, _glyph_w, _glyph_c = stacked_glyphs.shape
        mask_top_blank = torch.zeros((_glyph_b, total_glyph_h, _glyph_w), dtype=torch.float32, device=mask.device)
        
        # --- 4. 最终拼接 ---
        # 在最终拼接前，确保设备匹配
        stacked_glyphs = stacked_glyphs.to(image.device)
        
        # 将堆叠后的字形图块与原始图像拼接
        final_stacked_image = torch.cat((stacked_glyphs, image), dim=1)
        
        # 将纯黑的顶部遮罩与原始用户遮罩拼接
        final_stacked_mask = torch.cat((mask_top_blank, mask), dim=1)
        
        return (final_stacked_image, final_stacked_mask)


# ======================================================================
# 3. 工具节点 (保持不变)
# ======================================================================

class CropImageByReference:
    """
    根据参考图的高度，从待裁剪图片的底部裁剪出相同高度的区域。
    这用于在修复完成后，裁掉顶部的字形图，只保留场景部分。
    """
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "image_to_crop": ("IMAGE",), "reference_image": ("IMAGE",), }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "crop_image"
    CATEGORY = "Text Glyphs/Image Utils"

    def crop_image(self, image_to_crop, reference_image):
        _ref_b, ref_h, _ref_w, _ref_c = reference_image.shape
        _crop_b, crop_h, _crop_w, _crop_c = image_to_crop.shape
        
        if ref_h >= crop_h: 
            return (image_to_crop,)
            
        # 从底部裁剪：[-ref_h:] 意为取最后 ref_h 个像素
        return (image_to_crop[:, -ref_h:, :, :],)


# ======================================================================
# 节点注册
# ======================================================================
NODE_CLASS_MAPPINGS = {
    "GlyphInpaintPreprocessor": GlyphInpaintPreprocessor,
    "CropImageByReference": CropImageByReference,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GlyphInpaintPreprocessor": "字型内补预处理器 (多行)",
    "CropImageByReference": "按参考图裁剪",
}


# import torch
# import numpy as np
# from PIL import Image, ImageDraw, ImageFont
# import torch.nn.functional as F


# def draw_glyph_flexible(font, text, width, height, max_font_size=140):
#     if width <= 0: width = 1
#     height = min(width // 6, height)
#     img = Image.new(mode='1', size=(width, height), color=0)
#     if not text or not text.strip(): return img
#     draw = ImageDraw.Draw(img)
#     g_size = 50
#     new_font = font.font_variant(size=g_size)
#     try:
#         bbox = new_font.getbbox(text)
#         text_width_initial = max(bbox[2] - bbox[0], 1)
#         text_height_initial = max(bbox[3] - bbox[1], 1)
#     except AttributeError: # 兼容旧版 Pillow
#         text_width_initial, text_height_initial = draw.textsize(text, font=new_font)
#         text_width_initial = max(text_width_initial, 1)
#         text_height_initial = max(text_height_initial, 1)
#     width_ratio = width * 0.9 / text_width_initial
#     height_ratio = height * 0.9 / text_height_initial
#     ratio = min(width_ratio, height_ratio)
#     final_font_size = int(g_size * ratio)
#     if width > 1280: max_font_size = 180
#     if width > 2048: max_font_size = 280
#     final_font_size = min(final_font_size, max_font_size)
#     new_font = font.font_variant(size=max(final_font_size, 10))
#     draw.text((width / 2, height / 2), text, font=new_font, fill=1, anchor='mm')
#     return img


# # 节点 1
# class GlyphInpaintPreprocessor:
#     """
#     一个强大的预处理节点，它接收原始图像和遮罩，然后完成所有准备工作：
#     1. 生成字型图。
#     2. 拼接图像（字型在上，原图在下）。
#     3. 拼接遮罩（空白在上，原遮罩在下）。
#     最后输出可直接用于内补模型的图像和遮罩。
#     """
#     @classmethod
#     def INPUT_TYPES(s):
#         return {
#             "required": {
#                 "image": ("IMAGE",),
#                 "mask": ("MASK",),
#                 "text": ("STRING", {"multiline": True, "default": "场景文本编辑"}),
#                 "font_path": ("STRING", {"default": "custom_nodes/Arial-Unicode-Regular.ttf"}),
#                 "max_font_size": ("INT", {"default": 140, "min": 10, "max": 1024, "step": 1}),
#             }
#         }

#     RETURN_TYPES = ("IMAGE", "MASK")
#     RETURN_NAMES = ("stacked_image", "stacked_mask")
#     FUNCTION = "preprocess"
#     CATEGORY = "Text Glyphs"

#     def preprocess(self, image, mask, text, font_path, max_font_size):
#         # --- 1. 生成字型图 ---
#         _b, height, width, _c = image.shape
#         try: font = ImageFont.truetype(font_path, size=50)
#         except IOError: font = ImageFont.load_default()
        
#         glyph_pil = draw_glyph_flexible(font=font, text=text, width=width, height=height, max_font_size=max_font_size)
#         glyph_pil_rgb = glyph_pil.convert("RGB")
#         glyph_np = np.array(glyph_pil_rgb).astype(np.float32) / 255.0
#         image_glyph = torch.from_numpy(glyph_np).unsqueeze(0)
        
#         # --- 2. 拼接图像 ---
#         # 确保字型图和原图设备一致
#         image_glyph = image_glyph.to(image.device)
#         stacked_image = torch.cat((image_glyph, image), dim=1)

#         # --- 3. 拼接遮罩 ---
#         # `mask` 输入是 (B, H, W)
#         # `image_glyph` 是 (B, H_glyph, W, C)
#         _gb, gh, gw, _gc = image_glyph.shape
        
#         # 创建与字型图等大的空白遮罩 (纯黑)
#         mask_top_blank = torch.zeros((_gb, gh, gw), dtype=torch.float32, device=mask.device)
        
#         # 拼接遮罩
#         stacked_mask = torch.cat((mask_top_blank, mask), dim=1) # dim=1 是高度维度
        
#         return (stacked_image, stacked_mask)


# # 节点 2: 【可选】按参考图裁剪 (CropImageByReference)
# class CropImageByReference:
#     """
#     根据参考图片的高度，从待裁剪图片的底部裁剪出相同高度的区域。
#     """
#     @classmethod
#     def INPUT_TYPES(s):
#         return { "required": { "image_to_crop": ("IMAGE",), "reference_image": ("IMAGE",), }}

#     RETURN_TYPES = ("IMAGE",)
#     FUNCTION = "crop_image"
#     CATEGORY = "Text Glyphs/Image Utils"

#     def crop_image(self, image_to_crop, reference_image):
#         _ref_b, ref_h, _ref_w, _ref_c = reference_image.shape
#         _crop_b, crop_h, _crop_w, _crop_c = image_to_crop.shape
#         if ref_h > crop_h: return (image_to_crop,)
#         return (image_to_crop[:, -ref_h:, :, :],)

# # ======================================================================
# # 节点注册 (Node Registration)
# # ======================================================================
# NODE_CLASS_MAPPINGS = {
#     "GlyphInpaintPreprocessor": GlyphInpaintPreprocessor,
#     "CropImageByReference": CropImageByReference,
# }

# NODE_DISPLAY_NAME_MAPPINGS = {
#     "GlyphInpaintPreprocessor": "字型内补预处理器 (Glyph Preprocessor)",
#     "CropImageByReference": "按参考图裁剪 (Crop By Reference)",
# }