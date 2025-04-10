import itertools
from collections import defaultdict, Counter
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import textstat
from datasets import load_dataset
from scipy.stats import friedmanchisquare, kruskal
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.contingency_tables import cochrans_q, mcnemar
import pingouin as pg
from sympy import false

from config import Config, Credentials
from graphics.coverage_kde_distribution import create_coverage_kde_dist
from graphics.line_chart import create_line_chart, save_legend_only
from graphics.mcl_line_chart import create_mcl_multi_line_chart
from graphics.multi_line_chart import create_multi_line_chart
from graphics.wordnet_dist import create_wordnet_dist
from latex.create_context_table import generate_context_table
from latex.create_context_table_small import generate_context_table_small
from latex.create_dpo_table import generate_dpo_table
from latex.create_duplicate_wordnet_table import generate_duplicate_wordnet_table
from latex.create_hown_table import generate_hown_overview_table
from latex.create_mulit_lang_table import generate_multi_lang_table
from latex.create_multi_lang_avg_def_table import generate_multi_lang_avg_def_table
from latex.create_multi_lang_readability import generate_multi_lang_readability_table
from reader import JSONLineReader


def nested_dict():
    return defaultdict(nested_dict)


def to_dict(d):
    return {k: to_dict(v) for k, v in d.items()} if isinstance(d, defaultdict) else d


def diff_entry(ctx_val, base_val):
    return (ctx_val, ctx_val - base_val) if ctx_val is not None and base_val is not None else (ctx_val, None)


class Analysis:
    TYPES = ['normal', 'simple', 'child']
    LANGUAGES = ['en', 'fr', 'ar', 'ru', 'zh']
    MODELS = {
        'Llama 3.1 8B': 'llama-v3p1-8b-instruct',
        'DPO Llama 3.1 8B': 'dpo-llama-v3p1-8b-instruct',
        'GPT-4o mini': 'gpt-4o-mini',
        'Qwen3-30B A3B': 'qwen3-30b-a3b',
        'Llama 4 Maverick': 'llama4-maverick-instruct-basic',
        'DeepSeek v3': 'deepseek-v3',
    }

    def filter_hown_valid_words(self, stats):
        all_data = [
            {**result, 'prompt_type': prompt, 'model': model}
            for prompt in stats.keys()
            for model in self.MODELS
            for result in stats[prompt][model]
        ]

        df = pd.DataFrame(all_data)
        df['combo'] = df['prompt_type'] + ' | ' + df['model']
        expected = len(stats.keys()) * len(self.MODELS)
        valid_words = set(df.groupby('word')['combo'].nunique()[lambda x: x == expected].index)

        return {
            prompt: {
                model: [r for r in stats[prompt][model] if r['word'] in valid_words]
                for model in self.MODELS
            }
            for prompt in stats.keys()
        }

    def read_model_data(self, base_file: str) -> Dict[str, Dict[str, List[Dict]]]:
        stats = {}
        reader = JSONLineReader()
        for type_ in self.TYPES:
            stats[type_] = {
                name: reader.read(base_file.format(model=model_id, type=type_))
                for name, model_id in self.MODELS.items()
            }
        return self.filter_hown_valid_words(stats)

    def filter_mcl_valid_words(self, stats):
        cleaned_stats = {}
        expected = len(self.TYPES) * len(self.MODELS)

        for lang in self.LANGUAGES:
            all_data = [
                {**result, 'prompt_type': prompt, 'model': model}
                for prompt in self.TYPES
                for model in self.MODELS
                for result in stats[prompt][lang][model]
            ]

            df = pd.DataFrame(all_data)
            df['combo'] = df['prompt_type'] + ' | ' + df['model']
            valid_words = set(df.groupby('word')['combo'].nunique()[lambda x: x == expected].index)

            for prompt in self.TYPES:
                if prompt not in cleaned_stats:
                    cleaned_stats[prompt] = {}

                cleaned_stats[prompt][lang] = {
                    model: [
                        r for r in stats[prompt][lang][model]
                        if r['word'] in valid_words
                    ]
                    for model in self.MODELS
                }
        return cleaned_stats

    def read_model_lang_data(self, base_file: str) -> Dict[str, Dict[str, Dict[str, List[Dict]]]]:
        stats = {type_: {} for type_ in self.TYPES}
        reader = JSONLineReader()
        for type_ in self.TYPES:
            for lang in self.LANGUAGES:
                stats[type_][lang] = {
                    name: reader.read(base_file.format(model=model_id, type=type_, lang=lang))
                    for name, model_id in self.MODELS.items()
                }
        return self.filter_mcl_valid_words(stats)

    def calculate_completeness(self, results: List[Dict]) -> Dict[str, float]:
        counter = Counter((item['complete_marker'], item['coarse_synsets_covered'] == 1) for item in results)
        total = sum(counter.values())

        yes_true = counter.get((True, True), 0)
        yes_false = counter.get((True, False), 0)
        no_true = counter.get((False, True), 0)
        yes_any_true = yes_true + yes_false + no_true

        return {
            'complete': yes_any_true / total * 100,
            'both': yes_true / total * 100,
            'context': (yes_false + yes_true) / total * 100,
            'full': (no_true + yes_true) / total * 100,
        }

    def calculate_definition_count(self, results: List[Dict]) -> Dict[str, float]:
        counts = [len(r['definitions']) for r in results]
        multi_counts = [c for c in counts if c > 1]
        return {
            'avg_definitions': np.mean(counts),
            'avg_definitions_in_multi': np.mean(multi_counts) if multi_counts else 0,
        }

    def calculate_coarse_synset_coverage(self, results: List[Dict], filter_marker: bool = False) -> Optional[float]:
        if 'coarse_synsets_covered' not in results[0]:
            return None
        values = [
            item['coarse_synsets_covered']
            for item in results
            if not filter_marker or not item['complete_marker']
        ]
        return np.mean(values) * 100 if values else None

    def analyze_results(self, results: List[Dict], fk_grades: bool = True, lang: str = 'en', reading_score = 'fkgl') -> Dict[str, Any]:
        total = len(results)
        num_defs = [r['category'] for r in results]
        complete_markers = [r['complete_marker'] for r in results]
        fk_grade = None
        if fk_grades:
            if lang in ('en', 'fr', 'ru'):
                textstat.textstat.set_lang(lang)
            else:
                textstat.textstat.set_lang('en')

            if reading_score == 'fkgl':
                fk_grade = np.mean([textstat.textstat.flesch_kincaid_grade(r['model_response']) for r in results])
            else:
                fk_grade = np.mean([textstat.textstat.flesch_reading_ease(r['model_response']) for r in results])

        definition_counts = self.calculate_definition_count(results)
        coarse_synset_coverage = self.calculate_coarse_synset_coverage(results, filter_marker=False)

        if 'coarse_synsets_covered' in results[0]:
            counter = Counter((r['category'], r['complete_marker'], r['coarse_synsets_covered'] == 1) for r in results)
            joint_counter = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
            for (n_defs, context, covered), count in counter.items():
                joint_counter[n_defs][context][covered] = count
            completeness = self.calculate_completeness(results)
            multi_if_marker = sum(joint_counter.get('Multiple', {}).get(True, {}).values())
            multi_same_def = sum(
                any(v > 1 for k, v in Counter(r for r in d['wordnet_rankings'] if r != -1).items())
                for d in results
            )
        else:
            counter = Counter((r['category'], r['complete_marker']) for r in results)
            joint_counter = defaultdict(lambda: defaultdict(int))
            for (n_defs, context), count in counter.items():
                joint_counter[n_defs][context] = count
            completeness = None
            multi_same_def = None
            multi_if_marker = joint_counter.get('Multiple', {}).get(True, 0)

        sense_aware = sum(1 for d in results if d.get('category') == 'Multiple' or d.get('complete_marker') is True)
        complete_markers_distribution = dict(Counter(complete_markers))
        category_distribution = dict(Counter(num_defs))
        return {
            'total': total,
            'fk_grade': fk_grade,
            'definition_counts': definition_counts,
            'category_distribution': category_distribution,
            'complete_markers_distribution': complete_markers_distribution,
            'joint_counter': to_dict(joint_counter),
            'completeness': completeness,
            'coarse_synset_coverage': coarse_synset_coverage,
            'multi_if_marker': multi_if_marker / complete_markers_distribution.get(True) if complete_markers_distribution.get(True) else None,
            'sense_awareness': sense_aware / total * 100,
            'more_than_one': category_distribution.get('Multiple',  0) / total * 100,
            'multi_same_def': multi_same_def / category_distribution.get('Multiple',  0)  * 100 if multi_same_def and category_distribution.get('Multiple') else None,
        }

    def multi_lang_readability_table(self):
        stats = self.get_mclwic_stats(reading_score='fre')

        table_data = defaultdict(list)
        for type_, langs in stats.items():
            for lang, lang_data in langs.items():
                if lang not in ['en', 'fr', 'ru']:
                    continue
                for model, data in lang_data.items():
                    table_data[model].append(data.get('fk_grade'))

        generate_multi_lang_readability_table(table_data)


    def multi_lang_table(self):
        stats = self.get_mclwic_stats()

        table_data = defaultdict(lambda: defaultdict(list))
        for type_, langs in stats.items():
            for model in self.MODELS.keys():
                def get_percentage(key, lang):
                    return langs[lang][model]['complete_markers_distribution'].get(key, 0) / langs[lang][model]['total'] * 100

                percentages_sense_aware = []
                percentages_multi = []
                percentages_hesa = []
                for lang in self.LANGUAGES:
                    percentage_multi = langs[lang][model]['category_distribution'].get('Multiple', 0) / langs[lang][model][
                        'total'] * 100
                    percentage_hesa = get_percentage(True, lang)
                    percentage_sense_aware = langs[lang][model]['sense_awareness']
                    percentages_sense_aware.append(percentage_sense_aware)
                    percentages_multi.append(percentage_multi)
                    percentages_hesa.append(percentage_hesa)

                table_data[type_][model].extend(percentages_sense_aware)
                table_data[type_][model].extend(percentages_multi)
                table_data[type_][model].extend(percentages_hesa)

        generate_multi_lang_table(table_data)

    def multi_lang_avg_def_table(self):
        stats = self.get_mclwic_stats()

        table_data = defaultdict(list)
        for type_, langs in stats.items():
            for lang, lang_data in langs.items():
                for model, data in lang_data.items():
                    table_data[model].append(data.get('definition_counts').get('avg_definitions_in_multi'))

        generate_multi_lang_avg_def_table(table_data)

    def filter_valid_words_context(self, data, ctx_data):
        combined = {
            **data,
            **{f"{k}_ctx": v for k, v in ctx_data.items()}
        }
        filtered = self.filter_hown_valid_words(combined)
        data = {k: v for k, v in filtered.items() if not k.endswith('_ctx')}
        ctx_data = {k[:-4]: v for k, v in filtered.items() if k.endswith('_ctx')}
        return data, ctx_data

    def context_table(self):
        data = self.get_hown_results()
        ctx_data = self.get_hown_results(w_context=True)

        data, ctx_data = self.filter_valid_words_context(data, ctx_data)

        stats = nested_dict()
        for type_ in data:
            for model in data[type_]:
                stats[type_][model] = {
                    'base': self.analyze_results(data[type_][model]),
                    'ctx': self.analyze_results(ctx_data[type_][model]),
                }

        table_data = defaultdict(lambda: defaultdict(list))
        table_data_small = defaultdict(lambda: defaultdict(list))
        for type_, models in stats.items():
            for model, res in models.items():
                fkgl, fkgl_ctx = res['base']['fk_grade'], res['ctx']['fk_grade']
                c, c_ctx = res['base']['completeness'], res['ctx']['completeness']
                s, s_ctx = res['base']['coarse_synset_coverage'], res['ctx']['coarse_synset_coverage']
                sa, sa_ctx = res['base']['sense_awareness'], res['ctx']['sense_awareness']
                md, md_ctx = (res['base']['category_distribution'].get('Multiple') / res['base']['total'] * 100,
                              res['ctx']['category_distribution'].get('Multiple') / res['ctx']['total'] * 100)

                table_data[type_][model] = [
                    diff_entry(fkgl_ctx, fkgl),
                    diff_entry(sa_ctx, sa),
                    diff_entry(md_ctx, md),
                    diff_entry(c_ctx.get('context'), c.get('context')),
                    diff_entry(c_ctx.get('full'), c.get('full')),
                    diff_entry(c_ctx.get('both'), c.get('both')),
                ]

                table_data_small[type_][model] = [
                    diff_entry(c_ctx.get('complete'), c.get('complete')),
                    diff_entry(s_ctx, s) if s and s_ctx else (s_ctx, None),
                ]

        generate_context_table(table_data)
        generate_context_table_small(table_data_small)

    def get_hown_results(self, w_context: bool = False):
        base_file = 'batches/homonymy-high-freq/{model}/homonymy-high-freq-{model}-output-judge-{type}'
        if w_context:
            base_file += "_w_context"
        base_file += "_en-parsed-raw.jsonl"
        return self.read_model_data(base_file)

    def get_hown_stats(self, w_context: bool = False):
        data = self.get_hown_results(w_context=w_context)

        stats = nested_dict()
        for type_ in data:
            for model in data[type_]:
                if model == 'DPO Llama 3.1 8B':
                    continue
                stats[type_][model] = self.analyze_results(data[type_][model])

        return stats

    def get_mclwic_results(self):
        base_file = 'batches/mcl-wic/{model}/mcl-wic-{model}-output-judge-{type}_{lang}-parsed-raw.jsonl'
        return self.read_model_lang_data(base_file)

    def get_mclwic_stats(self, reading_score='fre'):
        data = self.get_mclwic_results()

        stats = nested_dict()
        for type_, langs in data.items():
            for lang, models in langs.items():
                for model, results in models.items():
                    stats[type_][lang][model] = self.analyze_results(results, lang=lang, reading_score=reading_score)
        return stats

    def hown_table(self):
        stats = self.get_hown_stats()

        table_data = defaultdict(lambda: defaultdict(list))
        for type_, models in stats.items():
            for model, res in models.items():
                completeness = res['completeness']
                table_data[type_][model] = [
                    res['fk_grade'],
                    res['sense_awareness'],
                    res['more_than_one'],
                    completeness['context'],
                    completeness['full'],
                    completeness['both'],
                    completeness['complete'],
                    res['coarse_synset_coverage'],
                ]
                print(res['multi_if_marker'])
        generate_hown_overview_table(table_data)

    def hown_completness_graph(self):
        stats = self.get_hown_stats()

        table_data = defaultdict(list)
        for type_, models in stats.items():
            for model, res in models.items():
                completeness = res['completeness']
                table_data[model].append(completeness['complete'])

        create_line_chart(table_data)

    def mclwic_lang_overview_graph(self):
        stats = self.get_mclwic_stats()

        table_data = defaultdict(lambda: defaultdict(list))
        for type_, langs in stats.items():
            for lang in self.LANGUAGES:
                percentage_ones = []
                complete_markers = []
                for model in self.MODELS.keys():
                    percentage_ones.append(langs[lang][model]['category_distribution'].get('One', 0) / langs[lang][model]['total'])
                    complete_markers.append(langs[lang][model]['complete_markers_distribution'].get(True, 0) / langs[lang][model]['total'])

                table_data['one_def'][lang].append(np.mean(percentage_ones) * 100)
                table_data['complete_marker'][lang].append(np.mean(complete_markers) * 100)

        create_multi_line_chart(table_data, 'language')
        create_mcl_multi_line_chart(table_data, 'language', 'complete')
        create_mcl_multi_line_chart(table_data, 'language', 'def')

    def mclwic_model_overview_graph(self):
        data = self.get_mclwic_results()

        stats = nested_dict()
        for type_, langs in data.items():
            all_lang_results = defaultdict(list)
            for lang, models in langs.items():
                for model, results in models.items():
                    all_lang_results[model].extend(results)

            for model, results in all_lang_results.items():
                stats[type_][model] = self.analyze_results(results, fk_grades=False, reading_score='fre')

        table_data = defaultdict(lambda: defaultdict(list))
        for type_, models in stats.items():
            for model in self.MODELS.keys():
                table_data['one_def'][model].append(models[model]['category_distribution'].get('One', 0) / stats[type_][model]['total'] * 100)
                table_data['complete_marker'][model].append(models[model]['complete_markers_distribution'].get(True, 0) / stats[type_][model]['total'] * 100)

        create_multi_line_chart(table_data, 'model')
        create_mcl_multi_line_chart(table_data, 'model', 'complete')
        create_mcl_multi_line_chart(table_data, 'model', 'def')

    def mclwic_lang_models_graph(self):
        data = self.get_mclwic_stats()

        stats = nested_dict()
        for type_, langs in data.items():
            for lang, models in langs.items():
                for model, results in models.items():
                    stats[lang][model][type_] = results

        for lang, models in stats.items():
            table_data = defaultdict(list)
            for model, model_data in models.items():
                for type_, results in model_data.items():
                    table_data[model].append(results['sense_awareness'])

            create_line_chart(table_data, f'sense-awareness-{lang}.pdf', '', 'Sense Aware (%)', show_legend=false)
            save_legend_only(table_data, f'sense-awareness-legend.pdf')

    def wordnet_dist(self):
        data = self.get_hown_results()

        for type_, model_data in data.items():
            rank_counts_by_model = defaultdict(Counter)
            all_ranks = set()

            for model, data in model_data.items():
                for result in data:
                    if result['coarse_synsets_covered'] < 1:
                        for rank in set(result.get('wordnet_rankings', [])):
                            if rank != -1:
                                rank += 1
                                rank_counts_by_model[model][rank] += 1
                                all_ranks.add(rank)

            create_wordnet_dist(all_ranks, rank_counts_by_model, type_)

    def coverage_kde_distribution(self):
        data = self.get_hown_results()

        for type_, model_data in data.items():
            models_results = {}
            for model, data in model_data.items():
                coarse_synsets_covered = [
                    item['coarse_synsets_covered']
                    for item in data
                    if item['complete_marker'] == False
                ]
                models_results[model] = coarse_synsets_covered
            create_coverage_kde_dist(models_results, type_)

    def hown_statistical_difference(self):
        def calculate_categories(row):
            complete_marker = row['complete_marker']
            coarse_synsets_covered = row['coarse_synsets_covered']

            complete = True if complete_marker else False
            both = True if complete_marker and coarse_synsets_covered == 1 else False
            context = True if complete_marker and coarse_synsets_covered == 0 else False
            full = True if not complete_marker and coarse_synsets_covered == 1 else False

            return pd.Series({
                'word': row['word'],
                'prompt_type': row['prompt_type'],
                'category': 0 if row['category'] == 'One' else 1,
                'coarse_synsets_covered': row['coarse_synsets_covered'],
                'complete': complete,
                'both': both,
                'context': context,
                'full': full
            })

        categories = ['category', 'complete', 'full', 'context', 'both']

        data = self.get_hown_results()
        for model in self.MODELS.keys():
            print(50*'-')
            print(model)
            child = [{**result, 'prompt_type': 'child'} for result in data['child'][model]]
            simple = [{**result, 'prompt_type': 'simple'} for result in data['simple'][model]]
            normal = [{**result, 'prompt_type': 'normal'} for result in data['normal'][model]]

            df = pd.DataFrame(child + simple + normal)
            df = df.apply(calculate_categories, axis=1)

            counts = df.groupby('word')['prompt_type'].nunique()
            valid_words = counts[counts == 3].index
            df = df[df['word'].isin(valid_words)]

            for category in categories:
                print(50*'-')
                print(category)
                pivot_data = df.pivot(index='word', columns='prompt_type', values=category)
                # Ensure stat is numeric (0/1)
                pivot_data = pivot_data.astype(int)

                # Step 2: Perform Cochran’s Q test
                cochran_result = cochrans_q(pivot_data)
                stat = cochran_result.statistic
                p_value = cochran_result.pvalue
                print(f"Cochran’s Q Test:")
                print(f"Statistic: {stat:.4f}, P-value: {p_value:.4f}")

                # Step 3: Interpret results
                if p_value < 0.05:
                    print("Significant differences detected across prompts.")
                else:
                    print("No significant differences across prompts.")

                # Step 4: Post-hoc pairwise McNemar tests (if significant)
                if p_value < 0.05:
                    print("\nRunning post-hoc McNemar tests...")
                    pairs = list(itertools.combinations(pivot_data.columns, 2))
                    alpha = 0.05 / len(pairs)  # Bonferroni correction
                    for prompt1, prompt2 in pairs:
                        # Create contingency table for McNemar test
                        table = pd.crosstab(pivot_data[prompt1], pivot_data[prompt2])
                        # Ensure table is 2x2 (fill missing cells with 0 if needed)
                        table = table.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
                        # Run McNemar test
                        mcnemar_result = mcnemar(table, exact=False, correction=True)
                        p_mcnemar = mcnemar_result.pvalue
                        print(f"{prompt1} vs {prompt2}: p-value = {p_mcnemar:.4f}, Significant: {p_mcnemar < alpha}")

            pivot_float = df.pivot(index='word', columns='prompt_type', values='coarse_synsets_covered')

            # Step 1: Check normality
            normality = pg.normality(pivot_float)
            print("Normality Test (Shapiro-Wilk):")
            print(normality)

            # Step 2: Mauchly’s test for sphericity
            sphericity = pg.sphericity(df, dv='coarse_synsets_covered', within='prompt_type', subject='word')
            print("\nMauchly’s Test for Sphericity:")
            print(f"W = {sphericity.W:.4f}, p-value = {sphericity.pval:.4f}")

            # Step 3: Choose and run test
            if sphericity.pval > 0.05 and normality['pval'].min() > 0.05:
                # Sphericity and normality hold: Use Repeated Measures ANOVA
                print("\nRunning Repeated Measures ANOVA...")
                anova = AnovaRM(df, depvar='coarse_synsets_covered', subject='word', within=['prompt_type'])
                result = anova.fit()
                print(result)
                p_value = result.anova_table['Pr > F'][0]
                test_type = "ANOVA"
            else:
                # Sphericity or normality violated: Use Friedman test
                print("\nRunning Friedman Test...")
                stat, p_value = friedmanchisquare(pivot_float['child'], pivot_float['simple'], pivot_float['normal'])
                print(f"Friedman chi-squared statistic: {stat:.4f}, P-value: {p_value:.4f}")
                test_type = "Friedman"

            # Step 4: Interpret results
            if p_value < 0.05:
                print(f"Significant differences detected across prompts (p = {p_value:.4f}).")
            else:
                print(f"No significant differences across prompts (p = {p_value:.4f}).")

            # Step 5: Post-hoc tests (if significant)
            if p_value < 0.05:
                print("\nRunning post-hoc pairwise comparisons...")
                posthoc = pg.pairwise_tests(
                    data=df, dv='coarse_synsets_covered', within='prompt_type', subject='word',
                    parametric=(test_type == "ANOVA"), padjust='bonf'
                )
                print(posthoc[['A', 'B', 'T' if test_type == "ANOVA" else 'W-val', 'p-corr', 'p-adjust']])

    def get_dpo_results(self):
        stats_dpo, stats, qwen_stats = {}, {}, {}
        all_data = []
        reader = JSONLineReader()

        base_file = 'batches/homonymy-high-freq/dpo-llama-v3p1-8b-instruct/homonymy-high-freq-dpo-llama-v3p1-8b-instruct-output-judge-{type}_en-parsed-raw.jsonl'
        for type_ in self.TYPES:
            data = reader.read(base_file.format(type=type_))
            stats_dpo[type_] = data
            all_data.extend(data)

        base_file = 'batches/homonymy-high-freq/llama-v3p1-8b-instruct/homonymy-high-freq-llama-v3p1-8b-instruct-output-judge-{type}_en-parsed-raw.jsonl'
        for type_ in self.TYPES:
            data = reader.read(base_file.format(type=type_))
            stats[type_] = data
            all_data.extend(data)

        base_file = 'batches/homonymy-high-freq/qwen3-30b-a3b/homonymy-high-freq-qwen3-30b-a3b-output-judge-{type}_en-parsed-raw.jsonl'
        for type_ in self.TYPES:
            data = reader.read(base_file.format(type=type_))
            qwen_stats[type_] = data
            all_data.extend(data)

        df = pd.DataFrame(all_data)
        expected = 9
        word_counts = df['word'].value_counts()
        valid_words = word_counts[word_counts == expected].index

        stats_dpo = {
            type_: [entry for entry in entries if entry['word'] in valid_words]
            for type_, entries in stats_dpo.items()
        }
        stats = {
            type_: [entry for entry in entries if entry['word'] in valid_words]
            for type_, entries in stats.items()
        }
        qwen_stats = {
            type_: [entry for entry in entries if entry['word'] in valid_words]
            for type_, entries in stats.items()
        }

        return stats_dpo, stats, qwen_stats

    def dpo_table(self):
        results_dpo, results, qwen_results = self.get_dpo_results()
        dpo_dataset = load_dataset(Config.DATASETS['homonymy-dpo'], token=Credentials.hf_api_key)['train'].to_list()
        dpo_words = {entry['word'] for entry in dpo_dataset}

        stats_without_dpo = {}
        for type_ in self.TYPES:
            filtered_results_dpo = [entry for entry in results_dpo[type_] if entry['word'] not in dpo_words]
            filtered_results = [entry for entry in results[type_] if entry['word'] not in dpo_words]
            filtered_qwen_results = [entry for entry in qwen_results[type_] if entry['word'] not in dpo_words]
            stats_without_dpo[type_] = {
                'dpo': self.analyze_results(filtered_results_dpo),
                'normal': self.analyze_results(filtered_results),
                'qwen': self.analyze_results(filtered_qwen_results),
            }

        table_data = defaultdict(list)

        metrics = ['FKGL', 'Sense Aware', 'Multi. Def.', 'HeSA', 'Full', 'Both', 'Complete', 'Covered']

        for type_, res in stats_without_dpo.items():
            c, c_dpo = res['normal']['completeness'], res['dpo']['completeness']
            s, s_dpo = res['normal']['coarse_synset_coverage'], res['dpo']['coarse_synset_coverage']
            fk, fk_dpo = res['normal']['fk_grade'], res['dpo']['fk_grade']
            sa, sa_dpo = res['normal']['sense_awareness'], res['dpo']['sense_awareness']
            md, md_dpo = res['normal']['more_than_one'], res['dpo']['more_than_one']

            table_data[type_] = [
                diff_entry(fk_dpo, fk),
                diff_entry(sa_dpo, sa),
                diff_entry(md_dpo, md),
                diff_entry(c_dpo.get('context'), c.get('context')),
                diff_entry(c_dpo.get('full'), c.get('full')),
                diff_entry(c_dpo.get('both'), c.get('both')),
                diff_entry(c_dpo.get('complete'), c.get('complete')),
                diff_entry(s_dpo, s) if s and s_dpo else (s_dpo, None),
            ]
        table_data_transposed = defaultdict(list)
        for i, metric in enumerate(metrics):
            for type_ in table_data:
                table_data_transposed[metric].append(table_data[type_][i])

        for type_ in stats_without_dpo.keys():
            print(stats_without_dpo[type_]['qwen'])
        generate_dpo_table(table_data_transposed)

    def analyze_popularity_by_category(self):
        results = self.get_hown_results()

        category_frequencies = defaultdict(lambda: defaultdict(list))
        for type_, model_data in results.items():
            for model, data in model_data.items():
                for result in data:
                    avg_freq = result["avg_google_ngrams_frequency"]
                    complete = result["complete_marker"] or result["coarse_synsets_covered"] == 1
                    if avg_freq:
                        category_frequencies[type_][complete].append(avg_freq)

        print("Average frequency (% of total words) by category:")
        for type_, data in category_frequencies.items():
            print(50*'-')
            print(type_)

            for category, freqs in data.items():
                if freqs:
                    mean_freq = sum(freqs) / len(freqs)
                    count = len(freqs)
                    print(f"  {category} (n={count}): Mean={mean_freq:.6f}%")
                else:
                    print(f"  {category} (n=0): No data")

            valid_groups = [freqs for freqs in category_frequencies[type_].values() if len(freqs) > 1]
            if len(valid_groups) >= 2:
                try:
                    f_stat, p_value = kruskal(*valid_groups)
                    print(f"\nKruskal Test for frequency differences across categories:")
                    print(f"F-statistic: {f_stat:.4f}, p-value: {p_value:.4f}")
                    if p_value < 0.05:
                        print("Significant difference in popularity between categories (p < 0.05)")
                    else:
                        print("No significant difference in popularity between categories (p >= 0.05)")
                except Exception as e:
                    print(f"Error computing Kruskal: {e}")
            else:
                print("\nInsufficient data for Kruskal (need at least two categories with multiple samples)")

    def duplicate_wordnet_table(self):
        data = self.get_hown_stats()

        table_data = defaultdict(list)
        for type_, data in data.items():
            for model, data in data.items():
                table_data[model].append(data.get('multi_same_def'))
        generate_duplicate_wordnet_table(table_data)


if __name__ == "__main__":
    Analysis().mclwic_lang_models_graph()
