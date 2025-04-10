import matplotlib.pyplot as plt

completeness = {
    "Llama 3.1 8B": [45.73, 21.95, 27.44],
    "GPT-4o mini": [73.17, 19.51, 1.22],
    "Qwen3-30B A3B": [72.56, 31.71, 15.24],
    "Llama 4 Maverick": [52.44, 27.44, 33.54],
    "DeepSeek v3": [47.56, 21.95, 1.83],
}

def create_line_chart(data, output_file='completeness-by-model.pdf', xlabel='Prompt Type', ylabel='Completeness (%)', show_legend=True):
    # Data
    models = [
        "Llama 3.1 8B",
    "GPT-4o mini", "Qwen3-30B A3B",
        "Llama 4 Maverick", "DeepSeek v3", "DPO Llama 3.1 8B",
    ]
    prompts = ["Normal", "Simple", "ELI5"]

    markers = ['o', 's', '^', 'D', 'v', 'P']

    plt.figure(figsize=(10, 10)) # (16, 10)
    ax = plt.gca()

    for i, model in enumerate(models):
        ax.plot(
            prompts,
            data[model],
            label=model,
            marker=markers[i],
            markersize=14,
            linewidth=3
        )

    ax.set_xlabel(xlabel, fontsize=40)
    ax.set_ylabel(ylabel, fontsize=40)
    ax.tick_params(axis='both', labelsize=36, pad=12)

    ax.set_ylim(0, 100)
    ax.set_yticks([i * 10 for i in range(11)])
    ax.set_xticks(prompts)

    # Cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if show_legend:
        ax.legend(fontsize=26)
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches="tight")


def save_legend_only(data, output_file='legend.pdf'):
    models = [
        "Llama 3.1 8B",
    "GPT-4o mini", "Qwen3-30B A3B",
        "Llama 4 Maverick", "DeepSeek v3", "DPO Llama 3.1 8B",
    ]
    prompts = ["Normal", "Simple", "ELI5"]
    markers = ['o', 's', '^', 'D', 'v', 'P']

    fig, ax = plt.subplots()

    # Create dummy lines for legend
    handles = []
    for i, model in enumerate(models):
        line, = ax.plot([], [],
                        label=model,
                        marker=markers[i % len(markers)],
                        markersize=14,
                        linewidth=3)
        handles.append(line)

    # Create separate legend
    legend = ax.legend(handles=handles, fontsize=26, frameon=False, loc='center')

    # New figure for just the legend
    fig_legend = plt.figure(figsize=(2, 10))
    fig_legend.legend(handles=handles, labels=models, fontsize=26, loc='center', ncol=1)
    fig_legend.tight_layout()
    fig_legend.savefig(output_file, bbox_inches='tight', transparent=True)
