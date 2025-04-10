import matplotlib.pyplot as plt
import seaborn
from matplotlib.lines import Line2D

# Your data (unchanged)
data = {
    'one_def': {
        'DeepSeek v3': [14.32, 45.64, 86.91],
        'GPT-4o mini': [23.04, 57.1, 90.96],
        'Llama 3.1 8B': [71.86, 84.31, 94.71],
        'Llama 4 Maverick': [42.09, 63.57, 89.04],
        'Qwen3-30B A3B': [11.52, 38.57, 75.84]
    },
    'complete_marker': {
        'DeepSeek v3': [68.37, 13.64, 1.0],
        'GPT-4o mini': [53.99, 3.55, 0.5],
        'Llama 3.1 8B': [17.31, 1.43, 0.62],
        'Llama 4 Maverick': [36.24, 14.2, 1.56],
        'Qwen3-30B A3B': [76.96, 17.69, 2.8]
    }
}

def create_multi_line_chart(data, axis='model', lang=None):
    if axis == 'model':
        models = {
            "Llama 3.1 8B": "Llama 3.1 8B",
            #"DPO Llama 3.1 8B": "DPO Llama 3.1 8B",
            "GPT-4o mini": "GPT-4o mini",
            "Qwen3-30B A3B": "Qwen3-30B A3B",
            "Llama 4 Maverick": "Llama 4 Maverick",
            "DeepSeek v3": "DeepSeek v3",
        }
    elif axis == 'language':
        models = {
            "English": "en",
            "French": "fr",
            "Arabic": "ar",
            "Russian": "ru",
            "Chinese (Zh)": "zh"
        }


    prompts = ["Normal", "Simple", "ELI5"]

    markers = ['o', 's', '^', 'D', 'v', 'P']
    colors = seaborn.color_palette("colorblind", n_colors=len(models))

    plt.figure(figsize=(18, 10))
    ax = plt.gca()

    # Plot lines (unchanged)
    for i, kv in enumerate(models.items()):
        model, model_key = kv
        ax.plot(
            prompts,
            data['one_def'][model_key],
            marker=markers[i],
            markersize=14,
            linewidth=3,
            color=colors[i],
            linestyle='-'
        )
        ax.plot(
            prompts,
            data['complete_marker'][model_key],
            marker=markers[i],
            markersize=12,
            linewidth=3,
            color=colors[i],
            linestyle='--'
        )

    # Set labels and ticks (unchanged)
    ax.set_xlabel("Prompt Type", fontsize=40)
    ax.set_ylabel("Percentage (%)", fontsize=40)
    ax.tick_params(axis='both', labelsize=36, pad=12)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 110, 10))
    ax.set_xticks(prompts)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Create combined legend entries
    legend_handles = []
    legend_labels = []

    # Add model entries
    for i, model in enumerate(models):
        legend_handles.append(Line2D([0], [0], color=colors[i], marker=markers[i], linestyle='-', linewidth=2, markersize=10))
        legend_labels.append(model)

    legend_handles.append(Line2D([0], [0], color='none', marker='none', linestyle='none'))
    legend_labels.append('')

    # Add line type entries
    legend_handles.append(Line2D([0], [0], color='black', linestyle='-', linewidth=2))
    legend_labels.append('Single Definition')
    legend_handles.append(Line2D([0], [0], color='black', linestyle='--', linewidth=2))
    legend_labels.append('Complete Marker')


    # Create a single legend with two columns
    plt.legend(
        legend_handles,
        legend_labels,
        loc='upper left',  # Ensure it's anchored to the left side
        ncol=1,
        bbox_to_anchor=(1.01, 1),  # Move further to the right
        fontsize=26,
    )

    plt.tight_layout()
    plt.savefig(f'multi-overview-{axis}{f'-{lang}' if lang else ''}.png', dpi=300, bbox_inches="tight")
    plt.show()