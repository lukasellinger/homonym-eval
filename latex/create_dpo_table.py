from prompt_toolkit.input.vt100 import raw_mode
from pylatex import Document, Tabular, NoEscape, Table, Command
from pylatex.utils import bold

# Data structure for the table
dummy_data = {
    'FK Grade': [(78.05, 75.00), (6.10, 5.49), (57.32, 0.00)],
    'Complete':  [(78.05, 75.00), (6.10, 5.49), (57.32, 0.00)],
    'Full': [(78.05, 75.00), (6.10, 5.49), (57.32, 0.00)],
    'Context': [(78.05, 75.00), (6.10, 5.49), (57.32, 0.00)],
    'Both': [(78.05, 75.00), (6.10, 5.49), (57.32, 0.00)],
    'Covered': [(78.05, 75.00), (6.10, 5.49), (57.32, 0.00)],
}

def generate_dpo_table(data):
    doc = Document()
    with doc.create(Table(position='h!')) as table:
        table.append(NoEscape(r'\centering'))
        table.append(NoEscape(r'\small'))
        table.append(NoEscape(r'\setlength{\tabcolsep}{4pt}'))
        tabular = Tabular('l c c c', booktabs=True)
        tabular.add_row(
            [bold('Metric'), bold('ELI5'), bold('Simple'), bold('Normal')])
        tabular.add_hline()

        for metric, values in data.items():
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
            tabular.add_row([metric] + row_values)

        table.append(tabular)
        table.add_caption('Comparison of response completeness and readability across different multi-sense aware prompts.')

    doc.generate_tex('dpo-table')