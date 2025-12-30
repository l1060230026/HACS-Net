import matplotlib.pyplot as plt
import numpy as np

# Data setup
classes = ['Ceiling', 'Floor', 'Wall', 'Beam', 'Column', 'Window', 'Door', 'Table', 'Chair', 'Sofa', 'Bookcase']
# Data from your prompt
local_scores = [0.929, 0.966, 0.636, 0.006, 0.038, 0.236, 0.396, 0.619, 0.661, 0.356, 0.437]
global_scores = [0.934, 0.959, 0.660, 0.003, 0.005, 0.102, 0.326, 0.564, 0.625, 0.457, 0.529]

# Optional: You can include "Ours" (Combined) from Table 2 to show it beats both
ours_scores = [0.899, 0.962, 0.646, 0.032, 0.013, 0.261, 0.384, 0.609, 0.604, 0.711, 0.489] 

x = np.arange(len(classes))
width = 0.25  # width of the bars (adjusted for 3 groups)

fig, ax = plt.subplots(figsize=(14, 6))

# Create bars
rects1 = ax.bar(x - width, local_scores, width, label='Local Alignment Only (L1, L2)', color='#4e79a7', alpha=0.9, edgecolor='black', linewidth=0.5)
rects2 = ax.bar(x, global_scores, width, label='Global Alignment Only (L3)', color='#f28e2b', alpha=0.9, edgecolor='black', linewidth=0.5)
rects3 = ax.bar(x + width, ours_scores, width, label='Multi-level Alignment (Combined)', color='#59a14f', alpha=0.9, edgecolor='black', linewidth=0.5)

# Styling
ax.set_ylabel('IoU Score', fontsize=12, fontname='Times New Roman')
ax.set_title('Per-class Performance: Local vs. Global vs. Ours (Combined)', fontsize=14, fontname='Times New Roman', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=11, fontname='Times New Roman')
ax.legend(fontsize=11, loc='best')

# Grid
ax.yaxis.grid(True, linestyle='--', alpha=0.6)
ax.set_axisbelow(True)

# Layout adjustment
plt.tight_layout()

# Save the figure
plt.savefig('local_vs_global_vs_ours_comparison.png', dpi=300)
