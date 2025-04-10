from prompt_toolkit.input.vt100 import raw_mode
from pylatex import Document, Tabular, NoEscape, Table, Command
from pylatex.utils import bold

# Data structure for the table
dummy_data = {
    'child': {
        'Llama 3.1 8B': [(78.05, 75.00), (6.10, 5.49),
                         (57.32, 0.00), (14.63, 13.41),
                         (52.70, -17.29)],
    },
    'simple': {
        'Llama 3.1 8B': [(78.05, 75.00), (6.10, 5.49),
                         (57.32, 0.00), (14.63, 13.41),
                         (52.70, -17.29)],
    },
    'normal': {
        'Llama 3.1 8B': [(78.05, 75.00), (6.10, 5.49),
                         (57.32, 0.00), (14.63, 13.41),
                         (52.70, None)],
    }
}

def generate_context_table_small(data):
    doc = Document()
    with doc.create(Table(position='h!')) as table:
        table.append(NoEscape(r'\centering'))
        table.append(NoEscape(r'\small'))
        tabular = Tabular('l c c', booktabs=True)
        tabular.add_row(
            [bold('Prompt / Model'), bold('Complete'), bold('Covered')])

        for prompt, models in data.items():
            tabular.add_hline()
            if prompt == 'child':
                tabular.append(NoEscape(r'\multicolumn{3}{l}{\textbf{Prompt: ELI5}} \\'))
            elif prompt == 'simple':
                tabular.append(NoEscape(r'\multicolumn{3}{l}{\textbf{Prompt: Simple}} \\'))
            elif prompt == 'normal':
                tabular.append(NoEscape(r'\multicolumn{3}{l}{\textbf{Prompt: Normal}} \\'))

            for model, values in models.items():
                row_values = []
                for value in values:
                    raw_delta = value[1]
                    if raw_delta is not None:
                        color = 'own_grey'
                        delta_value = f'{raw_delta:.2f}'
                        if raw_delta > 0:
                            color = 'own_green'
                            delta_value = f'+{delta_value}'
                        elif raw_delta < 0:
                            color = 'own_red'
                        delta = r"\textcolor{" + color + "}{" + delta_value + "}"
                    else:
                        delta = ""
                    raw_value = value[0]
                    if raw_value is not None:
                        parsed_value = f'{raw_value:.2f}'
                    else:
                        parsed_value = ''
                    table_entry = f'{parsed_value} {delta}' if parsed_value and delta else '{---}'
                    row_values.append(NoEscape(table_entry))
                tabular.add_row([model] + row_values)

        table.append(tabular)
        table.add_caption('Comparison of response completeness and readability across different multi-sense aware prompts.')

    doc.generate_tex('context-table-small')