"""Stable legacy restoration instruction constants."""

INSTRUCTION_EN = (
    "Remove the visible image degradation and restore a clean, natural image. "
    "Preserve the primary subject, geometry, camera composition, background layout, "
    "colors, fine texture, text, logos, and small structural details."
)
INSTRUCTION_CN = (
    "去除图像中可见的退化并恢复干净、自然的图像；保持主体身份、几何结构、相机构图、"
    "背景布局、颜色、精细纹理、文字、标志和细小结构不变。"
)

__all__ = ["INSTRUCTION_EN", "INSTRUCTION_CN"]
