from pylatex import Document, NoEscape, Table, Tabular


def generate_multi_lang_avg_def_table(data):
    doc = Document()
    with doc.create(Table(position='h!')) as table:
        table.append(NoEscape(r'\centering'))
        table.append(NoEscape(r'\small'))
        table.append(NoEscape(r'\setlength{\tabcolsep}{4pt}'))

        with table.create(Tabular('l' + 'c' * 15, width=16, booktabs=True)) as tabular:
            table.append(NoEscape(r'\textbf{Model} & \multicolumn{3}{c}{\textbf{Normal}} & \multicolumn{3}{c}{\textbf{Simple}} & \multicolumn{3}{c}{\textbf{ELI5.}} \\'))
            tabular.add_hline(start=2, end=6, cmidruleoption='lr')
            tabular.add_hline(start=7, end=11, cmidruleoption='lr')
            tabular.add_hline(start=12, end=16, cmidruleoption='lr')

            tabular.add_row(['', 'En', 'Fr', 'Ar', 'Ru', 'Zh',
                             'En', 'Fr', 'Ar', 'Ru', 'Zh',
                             'En', 'Fr', 'Ar', 'Ru', 'Zh',])
            tabular.add_hline()
            for model, raw_data in data.items():
                data = [f'{r:.2f}' for r in raw_data]
                tabular.add_row([model] + data)
        table.add_caption('Scores per prompt type and language model')

    doc.generate_tex('multi-lang-avg-def-table')