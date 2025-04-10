from prompt_toolkit.input.vt100 import raw_mode
from pylatex import Document, Tabular, NoEscape, Table, Command
from pylatex.utils import bold


def generate_duplicate_wordnet_table(data):
    doc = Document()
    with doc.create(Table(position='h!')) as table:
        table.append(NoEscape(r'\centering'))
        table.append(NoEscape(r'\small'))
        table.append(NoEscape(r'\setlength{\tabcolsep}{4pt}'))
        tabular = Tabular('l c c c', booktabs=True)
        tabular.add_row(
            [bold('Model'), bold('ELI5'), bold('Simple'), bold('Normal')])
        tabular.add_hline()

        for model, values in data.items():
            row_values = [f'{raw_value:.2f}' for raw_value in values]
            tabular.add_row([model] + row_values)

        table.append(tabular)
        table.add_caption('Duplicate WordNet Table.')

    doc.generate_tex('duplicate_wordnet_table')