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

def generate_context_table(data):
    doc = Document()
    with doc.create(Table(position='h!')) as table:
        table.append(NoEscape(r'\centering'))
        table.append(NoEscape(r'\small'))
        tabular = Tabular('l c c c c c c c c c c c c', booktabs=True)

        tabular.append(NoEscape(
            r'\textbf{Model} & \multicolumn{2}{c}{\textbf{FKGL}} & \multicolumn{2}{c}{\textbf{Sense Aware}} & '
            r'\multicolumn{2}{c}{\textbf{Multi. Def}} & \multicolumn{2}{c}{\textbf{HeSA}} & '
            r'\multicolumn{2}{c}{\textbf{Full}} & \multicolumn{2}{c}{\textbf{Both}} \\'
        ))

        for prompt, models in data.items():
            tabular.add_hline()
            if prompt == 'child':
                tabular.append(NoEscape(r'\multicolumn{13}{l}{\textbf{Prompt: ELI5}} \\'))
            elif prompt == 'simple':
                tabular.append(NoEscape(r'\multicolumn{13}{l}{\textbf{Prompt: Simple}} \\'))
            elif prompt == 'normal':
                tabular.append(NoEscape(r'\multicolumn{13}{l}{\textbf{Prompt: Normal}} \\'))

            for model, values in models.items():
                row_values = [model]
                for val, delta in values:
                    val_str = f'{val:.2f}' if val is not None else ''
                    if delta is not None:
                        color = 'own_grey'
                        delta_str = f'{delta:.2f}'
                        if delta > 0:
                            color = 'own_green'
                            delta_str = f'+{delta_str}'
                        elif delta < 0:
                            color = 'own_red'
                        delta_str =  fr'\textcolor{{{color}}}{{\tablenum{{{delta_str}}}}}'
                    else:
                        delta_str = ''

                    row_values.extend([NoEscape(val_str), NoEscape(delta_str)])

                tabular.add_row(row_values)

        table.append(tabular)
        table.add_caption('Comparison of response completeness and readability across different multi-sense aware prompts.')

    doc.generate_tex('context-table')