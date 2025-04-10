import pandas as pd
from pylatex import Document, Tabular, NoEscape, Table, Command
from pylatex.utils import bold, italic

dummy_data = {
    'child': {
        'Llama 3.1 8B': [89.52, 94.47, 95.45, 97.27, 97.64,
                         1.50, 0.53, 0.65, 0.00, 0.39,
                         2.56, 3.05, 2.45, 3.89, 2.00]
    },
    'simple': {
        'Llama 3.1 8B': [89.52, 94.47, 95.45, 97.27, 97.64,
                         1.50, 0.53, 0.65, 0.00, 0.39,
                         2.56, 3.05, 2.45, 3.89, 2.00]
    },
    'normal': {
        'Llama 3.1 8B': [89.52, 94.47, 95.45, 97.27, 97.64,
                         1.50, 0.53, 0.65, 0.00, 0.39,
                         2.56, 3.05, 2.45, 3.89, 2.00]
    },
}

def generate_multi_lang_table(data):
    doc = Document()
    with doc.create(Table(position='h!')) as table:
        table.append(NoEscape(r'\centering'))
        table.append(NoEscape(r'\small'))
        table.append(NoEscape(r'\setlength{\tabcolsep}{4pt}'))

        with table.create(Tabular('l' + 'c' * 15, width=16, booktabs=True)) as tabular:
            table.append(NoEscape(r'\textbf{Prompt / Model} & \multicolumn{5}{c}{\textbf{Sense Aware}} & \multicolumn{5}{c}{\textbf{Multi. Def.}} & \multicolumn{5}{c}{\textbf{HeSA}} \\'))
            tabular.add_hline(start=2, end=6, cmidruleoption='lr')
            tabular.add_hline(start=7, end=11, cmidruleoption='lr')
            tabular.add_hline(start=12, end=16, cmidruleoption='lr')

            tabular.add_row(['', 'En', 'Fr', 'Ar', 'Ru', 'Zh',
                             'En', 'Fr', 'Ar', 'Ru', 'Zh',
                             'En', 'Fr', 'Ar', 'Ru', 'Zh'])

            for prompt, data in data.items():
                tabular.add_hline()
                if prompt == 'child':
                    tabular.append(NoEscape(r'\multicolumn{16}{l}{\textbf{Prompt: ELI5}} \\'))
                elif prompt == 'simple':
                    tabular.append(NoEscape(r'\multicolumn{16}{l}{\textbf{Prompt: Simple}} \\'))
                elif prompt == 'normal':
                    tabular.append(NoEscape(r'\multicolumn{16}{l}{\textbf{Prompt: Normal}} \\'))

                df = pd.DataFrame(data).T

                def highlight_max_and_second(s):
                    max_val = s.max()
                    second_max = s[s < max_val].max() if (s < max_val).any() else None
                    result = []
                    for v in s:
                        if v == max_val:
                            result.append(bold(f'{v:.2f}'))
                        elif v == second_max:
                            result.append(bold(f'{v:.2f}'))
                        else:
                            result.append(f'{v:.2f}')
                    return result
                highlighted_df = df.apply(highlight_max_and_second, axis=0)

                for model, values in highlighted_df.iterrows():
                    tabular.add_row([model] + values.tolist())
        table.add_caption('Evaluation scores per prompt type and language')

    doc.generate_tex('multi-lang-table')
