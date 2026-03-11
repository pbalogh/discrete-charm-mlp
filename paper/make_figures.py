"""Generate paper figures for The Discrete Charm of the MLP."""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 11

# === Figure 1: Two Regimes (dual-axis) ===
consensus = [0, 1, 2, 3, 4, 5, 6, 7]
fire_rate = [94.7, 83.6, 56.5, 24.4, 10.3, 4.5, 1.4, 0.5]
avg_norm  = [194.1, 155.5, 126.1, 102.4, 88.6, 82.3, 76.2, 70.0]

fig, ax1 = plt.subplots(figsize=(6, 3.8))

color1 = '#d62728'  # red
color2 = '#1f77b4'  # blue

ax1.set_xlabel('Default-ON consensus neurons firing (out of 7)', fontsize=12)
ax1.set_ylabel('N2123 fire rate (%)', color=color1, fontsize=12)
line1 = ax1.plot(consensus, fire_rate, 'o-', color=color1, linewidth=2.5, markersize=8, label='N2123 fire rate')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim(-5, 100)
ax1.set_xticks(consensus)

ax2 = ax1.twinx()
ax2.set_ylabel('MLP output norm', color=color2, fontsize=12)
line2 = ax2.plot(consensus, avg_norm, 's--', color=color2, linewidth=2.5, markersize=8, label='MLP output norm')
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim(50, 210)

# Combined legend
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right', fontsize=10, framealpha=0.9)

# Annotations
ax1.annotate('Consensus\nbreakdown', xy=(0, 94.7), xytext=(1.8, 82),
            fontsize=9, ha='center', color=color1,
            arrowprops=dict(arrowstyle='->', color=color1, lw=1.5))
ax1.annotate('Full\nagreement', xy=(7, 0.4), xytext=(5.5, 15),
            fontsize=9, ha='center', color=color1,
            arrowprops=dict(arrowstyle='->', color=color1, lw=1.5))

fig.tight_layout()
plt.savefig('two_regimes.pdf', bbox_inches='tight', dpi=300)
plt.savefig('two_regimes.png', bbox_inches='tight', dpi=300)
print("Saved two_regimes.pdf/.png")

# === Figure 2: Exception handler schematic ===
fig2, ax = plt.subplots(figsize=(6, 4))
ax.set_xlim(0, 10)
ax.set_ylim(-0.3, 7.5)
ax.axis('off')

# Box dimensions — symmetric layout
bw, bh = 3.5, 1.8  # box width, height
gap = 1.0  # gap between top boxes
top_y = 5.0
bot_y = 1.2

# Top row: centered pair
left_x = (10 - 2*bw - gap) / 2
right_x = left_x + bw + gap
left_cx = left_x + bw/2
right_cx = right_x + bw/2

# Consensus neurons box
rect1 = matplotlib.patches.FancyBboxPatch((left_x, top_y), bw, bh, boxstyle="round,pad=0.2",
        facecolor='#e8f4fd', edgecolor='#1f77b4', linewidth=2)
ax.add_patch(rect1)
ax.text(left_cx, top_y + bh/2, '7 Consensus\nNeurons', ha='center', va='center',
        fontsize=11, fontweight='bold', color='#1f77b4')

# N2123 box
rect2 = matplotlib.patches.FancyBboxPatch((right_x, top_y), bw, bh, boxstyle="round,pad=0.2",
        facecolor='#fde8e8', edgecolor='#d62728', linewidth=2)
ax.add_patch(rect2)
ax.text(right_cx, top_y + bh/2, 'N2123\nException Handler',
        ha='center', va='center', fontsize=11, fontweight='bold', color='#d62728')

# Mutual exclusivity arrow — between boxes, label above
arrow_y = top_y + bh/2
ax.annotate('', xy=(right_x - 0.1, arrow_y), xytext=(left_x + bw + 0.1, arrow_y),
           arrowprops=dict(arrowstyle='<->', color='#666', lw=2, linestyle='--'))
ax.text((left_cx + right_cx)/2, top_y + bh + 0.25, '93–98%\nexclusive',
        ha='center', va='bottom', fontsize=8, color='#666')

# Bottom row: aligned under top boxes
rect3 = matplotlib.patches.FancyBboxPatch((left_x, bot_y), bw, bh, boxstyle="round,pad=0.2",
        facecolor='#e8fde8', edgecolor='#2ca02c', linewidth=2)
ax.add_patch(rect3)
ax.text(left_cx, bot_y + bh/2, 'Linear Path\n(~90% of tokens)',
        ha='center', va='center', fontsize=10, color='#2ca02c')

rect4 = matplotlib.patches.FancyBboxPatch((right_x, bot_y), bw, bh, boxstyle="round,pad=0.2",
        facecolor='#fff3e0', edgecolor='#e65100', linewidth=2)
ax.add_patch(rect4)
ax.text(right_cx, bot_y + bh/2, 'Full Nonlinear\n(~10% of tokens)',
        ha='center', va='center', fontsize=10, color='#e65100')

# Arrows down — centered under each top box
ax.annotate('', xy=(left_cx, bot_y + bh + 0.05), xytext=(left_cx, top_y - 0.05),
           arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=2))
ax.text(left_cx - 0.8, (top_y + bot_y + bh) / 2, 'agree',
        fontsize=9, color='#2ca02c', fontstyle='italic', ha='center')

ax.annotate('', xy=(right_cx, bot_y + bh + 0.05), xytext=(right_cx, top_y - 0.05),
           arrowprops=dict(arrowstyle='->', color='#e65100', lw=2))
ax.text(right_cx + 0.9, (top_y + bot_y + bh) / 2, 'disagree',
        fontsize=9, color='#e65100', fontstyle='italic', ha='center')

# Norm annotations
ax.text(left_cx, bot_y - 0.5, 'norm ≈ 70', ha='center', fontsize=9, color='#666')
ax.text(right_cx, bot_y - 0.5, 'norm ≈ 194', ha='center', fontsize=9, color='#666')

fig2.tight_layout()
plt.savefig('exception_handler.pdf', bbox_inches='tight', dpi=300)
plt.savefig('exception_handler.png', bbox_inches='tight', dpi=300)
print("Saved exception_handler.pdf/.png")
