"""
生成标签颜色映射图（颜色条）
显示每个语义标签对应的颜色
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import sys
import os

# 添加项目根目录到路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from data_utils.indoor3d_util import g_label2color

# 类别列表（按照标签索引顺序）
classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 
           'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter']

def generate_colormap(output_path='label_colormap.png', figsize=(4, 6)):
    """
    生成标签颜色映射图（简化版：只显示色块和类别名称）
    
    Args:
        output_path: 输出图片路径
        figsize: 图片大小
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    
    # 创建图例
    y_start = 0.96
    y_step = 0.075
    patch_width = 0.12
    patch_height = 0.05
    text_offset = 0.25
    
    for i, class_name in enumerate(classes):
        if i not in g_label2color:
            continue
            
        color_rgb = g_label2color[i]  # RGB值，范围[0, 255]
        color_normalized = [c / 255.0 for c in color_rgb]  # 归一化到[0, 1]
        
        # 创建颜色块
        y_pos = y_start - i * y_step
        patch = mpatches.Rectangle((0.05, y_pos - patch_height/2), 
                                   patch_width, patch_height,
                                   facecolor=color_normalized,
                                   edgecolor='black', linewidth=1.2,
                                   transform=ax.transAxes)
        ax.add_patch(patch)
        
        # 添加类别名称（只显示类别名）
        ax.text(0.05 + patch_width + text_offset, y_pos, class_name, 
                fontsize=14, va='center', transform=ax.transAxes,
                fontweight='normal')
    
    # 设置坐标范围
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"颜色映射图已保存到: {output_path}")
    plt.close()
    
    # 同时打印文本格式的颜色映射
    print("\n" + "="*70)
    print("标签颜色映射表（RGB值范围: 0-255）")
    print("="*70)
    print(f"{'Label':<8} {'Class Name':<15} {'R':<5} {'G':<5} {'B':<5}")
    print("-"*70)
    for i, class_name in enumerate(classes):
        if i in g_label2color:
            color_rgb = g_label2color[i]
            print(f"{i:<8} {class_name:<15} {color_rgb[0]:<5} {color_rgb[1]:<5} {color_rgb[2]:<5}")
    print("="*70)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Generate Label Colormap')
    parser.add_argument('--output', type=str, default='label_colormap.png',
                       help='Output image path [default: label_colormap.png]')
    parser.add_argument('--width', type=float, default=4,
                       help='Figure width [default: 4]')
    parser.add_argument('--height', type=float, default=6,
                       help='Figure height [default: 6]')
    
    args = parser.parse_args()
    
    generate_colormap(args.output, figsize=(args.width, args.height))

