import seaborn
from matplotlib import pyplot as plt


def create_coverage_kde_dist(models_results, title: str):
    if 'simple' in title:
        title = 'Simple'
    elif 'child' in title:
        title = 'ELI5'
    elif 'normal' in title:
        title = 'Normal'

    plt.figure(figsize=(16, 10))
    ax = plt.gca()

    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (1, 1))]
    colors = seaborn.color_palette("colorblind", n_colors=len(models_results))

    model_names = list(models_results.keys())

    for i, model_name in enumerate(model_names):
        values = models_results[model_name]

        seaborn.kdeplot(
            values,
            ax=ax,
            fill=False,
            linestyle=line_styles[i % len(line_styles)],
            label=model_name,
            color=colors[i % len(colors)],
            bw_method=0.3,  # 0.3,
            cut=0,
            clip=(0, 1),
            linewidth=5,
        )

        plt.xlim(0, 1)

    ax.set_ylim(0, 3.5)
    ax.set_xticks([i / 10 for i in range(0, 11)])
    ax.set_xticklabels([f"{int(x * 100)}" for x in ax.get_xticks()])

    ax.set_xlabel("Proportion of Coarse Synsets Covered (%)", fontsize=40)
    ax.set_ylabel("Density", fontsize=40)
    ax.set_title(title, fontsize=40)
    ax.tick_params(axis='both', labelsize=36)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend(fontsize=26)
    plt.tight_layout()
    plt.savefig(f'sense-coverage-all-{title.lower()}.pdf', bbox_inches="tight")