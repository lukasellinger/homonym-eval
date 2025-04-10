import numpy as np
import seaborn
from matplotlib import pyplot as plt


def create_wordnet_dist(all_ranks, rank_counts_by_model, title: str):
    if 'simple' in title:
        title = 'Simple'
    elif 'child' in title:
        title = 'ELI5'
    elif 'normal' in title:
        title = 'Normal'

    num_models = len(rank_counts_by_model)
    bar_width = 0.167
    x = np.arange(len(all_ranks))

    # Color and hatch settings
    colors = seaborn.color_palette("colorblind", n_colors=num_models)
    hatches = ['/', '\\', '|', '-', '+', 'x']  # Optional for B/W

    plt.figure(figsize=(18, 10))
    ax = plt.gca()

    # Plot grouped bars
    for i, (model_name, counts) in enumerate(rank_counts_by_model.items()):
        heights = [counts.get(rank, 0) for rank in all_ranks]
        ax.bar(
            x + i * bar_width,
            heights,
            width=bar_width,
            label=model_name,
            color=colors[i],
            hatch=hatches[i % len(hatches)],
            edgecolor='black'
        )

    ax.tick_params(axis='both', labelsize=40)
    # Axes and ticks
    ax.set_xlabel("WordNet Sense Rank", fontsize=40)
    ax.set_ylabel("Frequency", fontsize=40)
    ax.set_title(title, fontsize=40)
    ax.set_xticks(x + (num_models - 1) * bar_width / 2)
    ax.set_xticklabels([str(r) for r in all_ranks])
    ax.tick_params(axis='both', labelsize=30)

    # Legend
    ax.legend(fontsize=26, loc='upper right')

    # Styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 70)

    plt.tight_layout()
    plt.savefig(f"sense-distribution-all-{title.lower()}.pdf", bbox_inches="tight")