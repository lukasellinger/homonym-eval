import re

import click
import questionary
from reader import JSONLineReader
import random
from sklearn.metrics import cohen_kappa_score

# Options for annotations
VALID_CATEGORY = ["None", "One", "Multiple"]
VALID_REMARK = ["True", "False"]
VALID_CONTEXT_REQ = ["True", "False"]


def sample_evaluations(evaluations: list[dict], sample_size: int) -> list[dict]:
    """Randomly sample up to sample_size evaluations."""
    if len(evaluations) <= sample_size:
        click.echo(f"Only {len(evaluations)} evaluations available; using all.")
        return evaluations
    random.seed(40)  # For reproducibility
    return random.sample(evaluations, sample_size)

def save_annotated_evaluations(annotated: list[dict], output_file: str):
    """Save human-annotated evaluations to JSONL."""
    JSONLineReader().write(output_file, annotated)

def calculate_correlation(annotated: list[dict]):
    """Calculate Cohen's Kappa for category and context_requested."""
    def extract(field: str, source: str = "automatic_evaluation") -> list:
        return [entry[source][field] for entry in annotated]

    def combine_fields(fields: list[str], source: str = "automatic_evaluation") -> list:
        return [any(entry[source][f] for f in fields) for entry in annotated]

    auto_num_defs = extract("category", "automatic_evaluation")
    human_num_defs = extract("category", "human_evaluation")

    auto_context = extract("context_clarification_request", "automatic_evaluation")
    human_context = extract("context_clarification_request", "human_evaluation")

    auto_remark = extract("remark_not_all_listed", "automatic_evaluation")
    human_remark = extract("remark_not_all_listed", "human_evaluation")

    auto_complete = combine_fields(["context_clarification_request", "remark_not_all_listed"], "automatic_evaluation")
    human_complete = combine_fields(["context_clarification_request", "remark_not_all_listed"], "human_evaluation")

    click.echo("\nCohen's Kappa for category:")
    try:
        kappa_defs = cohen_kappa_score(auto_num_defs, human_num_defs)
        click.echo(f"  Score: {kappa_defs:.5f}")
    except ValueError as e:
        click.echo(f"  Error calculating Kappa for definitions: {e}")

    click.echo("\nCohen's Kappa for complete marker:")
    try:
        kappa_context = cohen_kappa_score(auto_complete, human_complete, labels=[True, False])
        click.echo(f"  Score: {kappa_context:.5f}")
    except ValueError as e:
        click.echo(f"  Error calculating Kappa for context requested: {e}")

    click.echo("\nCohen's Kappa for context clarification request:")
    try:
        kappa_context = cohen_kappa_score(auto_context, human_context, labels=[True, False])
        click.echo(f"  Score: {kappa_context:.5f}")
    except ValueError as e:
        click.echo(f"  Error calculating Kappa for context requested: {e}")

    click.echo("\nCohen's Kappa for remark not all listed:")
    try:
        kappa_context = cohen_kappa_score(auto_remark, human_remark, labels=[True, False])
        click.echo(f"  Score: {kappa_context:.5f}")
    except ValueError as e:
        click.echo(f"  Error calculating Kappa for context requested: {e}")

@click.group()
def cli():
    """A simple app for annotating evaluations and calculating correlation."""
    pass

@cli.command()
@click.option('--input', required=True, help='Input JSONL file with evaluations.')
@click.option('--output', required=True, help='Output JSONL file for annotated evaluations.')
@click.option('--sample-size', default=20, type=int, help='Number of evaluations to sample.')
def annotate_responses(input, output, sample_size):
    """Annotate a sample of evaluations and calculate correlation."""
    # Load evaluations
    responses = JSONLineReader().read(input)
    click.echo(f"Loaded {len(responses)} evaluations.")

    # Sample evaluations
    sampled_evaluations = sample_evaluations(responses, sample_size)
    click.echo(f"Sampled {len(sampled_evaluations)} evaluations for annotation.")

    # Annotate each sample
    annotated_evaluations = []
    for i, eval in enumerate(sampled_evaluations, 1):
        click.echo(f"\n--- Annotating evaluation {i}/{len(sampled_evaluations)} ---")
        click.echo(f"Word: {eval['word']}")

        if 'model_response' in eval:
            model_response = eval['model_response']
        else:
            model_response = ''
        model_response = re.sub(r"<think>.*?</think>", "", model_response, flags=re.DOTALL).strip()

        click.echo(f"Model Response: {model_response}")

        # Annotate number of definitions
        category = questionary.select(
            "Select the category:",
            choices=VALID_CATEGORY
        ).ask()

        # Annotate context requested
        context_clarification = questionary.select(
            "Select if context was requested:",
            choices=VALID_CONTEXT_REQ
        ).ask()

        # Annotate context requested
        remark = questionary.select(
            "Select if there is a remark, that not all definitions are listed:",
            choices=VALID_REMARK
        ).ask()

        human_eval = {
            "category": category,
            "context_clarification_request": context_clarification == 'True',
            "remark_not_all_listed": remark == 'True',
        }

        annotated_evaluations.append({
            "word": eval.get("word"),
            "model_response": model_response,
            "human_evaluation": human_eval,
        })

    # Save annotated evaluations
    save_annotated_evaluations(annotated_evaluations, output)
    click.echo(f"Saved {len(annotated_evaluations)} annotated evaluations to {output}.")


@cli.command()
@click.option('--input', required=True, help='Input JSONL file with evaluations.')
@click.option('--output', required=True, help='Output JSONL file for annotated evaluations.')
@click.option('--sample-size', default=20, type=int, help='Number of evaluations to sample.')
def annotate(input, output, sample_size):
    """Annotate a sample of evaluations and calculate correlation."""
    # Load evaluations
    evaluations = JSONLineReader().read(input)
    click.echo(f"Loaded {len(evaluations)} evaluations.")

    # Sample evaluations
    sampled_evaluations = sample_evaluations(evaluations, sample_size)
    click.echo(f"Sampled {len(sampled_evaluations)} evaluations for annotation.")

    # Annotate each sample
    annotated_evaluations = []
    for i, eval in enumerate(sampled_evaluations, 1):
        click.echo(f"\n--- Annotating evaluation {i}/{len(sampled_evaluations)} ---")
        click.echo(f"Word: {eval['word']}")
        click.echo(f"Model Response: {eval['model_response']}")

        # Annotate number of definitions
        category = questionary.select(
            "Select the category:",
            choices=VALID_CATEGORY
        ).ask()

        # Annotate context requested
        context_clarification = questionary.select(
            "Select if context was requested:",
            choices=VALID_CONTEXT_REQ
        ).ask()

        # Annotate context requested
        remark = questionary.select(
            "Select if there is a remark, that not all definitions are listed:",
            choices=VALID_REMARK
        ).ask()

        human_eval = {
            "category": category,
            "context_clarification_request": context_clarification == 'True',
            "remark_not_all_listed": remark == 'True',
        }

        automatic_eval = {
            "category": eval.get("category"),
            "context_clarification_request": eval.get("context_clarification_request"),
            "remark_not_all_listed": eval.get("remark_not_all_listed"),
            "definitions": eval.get("definitions"),
        }

        annotated_evaluations.append({
            "word": eval.get("word"),
            "model_response": eval.get("model_response"),
            "automatic_evaluation": automatic_eval,
            "human_evaluation": human_eval,
            "avg_google_ngrams_frequency": eval.get("avg_google_ngrams_frequency"),
        })

    # Save annotated evaluations
    save_annotated_evaluations(annotated_evaluations, output)
    click.echo(f"Saved {len(annotated_evaluations)} annotated evaluations to {output}.")

    # Calculate correlation
    click.echo("\nAnalyzing correlation between automatic and human evaluations...")
    calculate_correlation(annotated_evaluations)

@cli.command()
@click.option('--file', 'files', multiple=True, required=True, help='One or more annotated JSONL files to calculate correlation.')
def calculate(files):
    """Calculate correlation from one or more annotated files."""
    reader = JSONLineReader()
    all_evaluations = []

    for file in files:
        evaluations = reader.read(file)
        all_evaluations.extend(evaluations)
        click.echo(f"Loaded {len(evaluations)} annotated evaluations from {file}.")

    click.echo(f"Total evaluations combined: {len(all_evaluations)}")
    calculate_correlation(all_evaluations)

@cli.command()
@click.option('--file', 'files', multiple=True, required=True, help='One or more annotated JSONL files to calculate correlation.')
@click.option('--human_file', 'human_files', multiple=True, required=True, help='One or more annotated JSONL files to calculate correlation.')
def calculate_human(files, human_files):
    """Calculate correlation from one or more annotated files."""
    reader = JSONLineReader()
    all_evaluations = []

    for human_file, file in zip(human_files, files):
        human_evaluations = reader.read(human_file)
        auto_evaluations = reader.read(file)
        auto_evaluations = {entry['word']: entry for entry in auto_evaluations}
        for evaluation in human_evaluations:
            auto_evaluation = auto_evaluations.get(evaluation['word'])
            if not auto_evaluation:
                continue
            all_evaluations.append({**evaluation, 'automatic_evaluation': {'category': auto_evaluation.get('category'),
                                                                           'context_clarification_request': auto_evaluation.get('context_clarification_request'),
                                                                           'remark_not_all_listed': auto_evaluation.get('remark_not_all_listed'),}})

    click.echo(f"Total evaluations combined: {len(all_evaluations)}")
    calculate_correlation(all_evaluations)

if __name__ == "__main__":
    #calculate(['batches/homonymy-high-freq/deepseek-v3/homonymy-high-freq-deepseek-v3-output-judge-simple_en-parsed.jsonl'],
    #          ['batches/homonymy-high-freq/deepseek-v3/homonymy-high-freq-responses-deepseek-v3-simple_en-human-annotate.jsonl'])
    cli()