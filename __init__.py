from .nodes import GlyphInpaintPreprocessor, CropImageByReference

NODE_CLASS_MAPPINGS = {
    "GlyphInpaintPreprocessor": GlyphInpaintPreprocessor,
    "CropImageByReference": CropImageByReference,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GlyphInpaintPreprocessor": "字型内补预处理器 (Glyph Preprocessor)",
    "CropImageByReference": "按参考图裁剪 (Crop By Reference)",
}

print("My Simplified Text Editing Nodes: Loaded successfully.")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']