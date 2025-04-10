from pylatex import NoEscape, Document, Table, Tabular


def generate_multi_lang_readability_table(data):
    doc = Document()
    with doc.create(Table(position='h!')) as table:
        table.append(NoEscape(r'\centering'))
        table.append(NoEscape(r'\small'))
        table.append(NoEscape(r'\setlength{\tabcolsep}{4pt}'))

        with table.create(Tabular('l' + 'c' * 9, width=10, booktabs=True)) as tabular:
            table.append(NoEscape(r'\textbf{Model} & \multicolumn{3}{c}{\textbf{Normal}} & \multicolumn{3}{c}{\textbf{Simple}} & \multicolumn{3}{c}{\textbf{ELI5.}} \\'))
            tabular.add_hline(start=2, end=4, cmidruleoption='lr')
            tabular.add_hline(start=5, end=7, cmidruleoption='lr')
            tabular.add_hline(start=8, end=10, cmidruleoption='lr')

            tabular.add_row(['', 'En', 'Fr', 'Ru',
                             'En', 'Fr', 'Ru',
                             'En', 'Fr', 'Ru',])
            tabular.add_hline()
            for model, raw_data in data.items():
                data = [f'{r:.2f}' for r in raw_data]
                tabular.add_row([model] + data)
        table.add_caption('Scores per prompt type and language model')

    doc.generate_tex('multi-lang-readability-table')
