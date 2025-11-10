# Simplifications are Absolutists
This repository contains the code and data for our paper:

**Simplifications are Absolutists: How Simplified Language Reduces Word Sense Awareness in LLM-Generated Definitions** <br>
see: [arxiv](https://arxiv.org/abs/2507.11981) or [proceedings](https://aclanthology.org/2025.ranlp-1.42)

> ⚠️ **Note**: The project structure is still a work in progress.

## 🚀 Additional Resources
### 📌 Extended Results and Materials
- All additional supporting materials and results are available here: [Extended Material](https://github.com/lukasellinger/homonym-eval/tree/main/extended-material.pdf)
  
### 🤖 Fine-tuned Model (DPO)
- Hugging Face: [lukasellinger/homonymy-dpo-llama-v3p1-8b-instruct](https://huggingface.co/lukasellinger/homonymy-dpo-llama-v3p1-8b-instruct)

### 📁 DPO Training Dataset (Preference Pairs)
- Hugging Face: [lukasellinger/homonymy-dpo](https://huggingface.co/datasets/lukasellinger/homonymy-dpo)

### 📊 Evaluation Datasets
- [HoWN Dataset](https://huggingface.co/datasets/lukasellinger/homonyms-with-wordnet-wsd)
- [ML-WiC Dataset](https://huggingface.co/datasets/ml-wic)
- 🔍 Evaluation input and results can be found in the `/batches` directory.

## ▶️ Evaluate Your Model
All necessary evaluation scripts can be found in `evaluation`.

1. **Set up configuration**  
   Copy the template configuration file and fill in your API keys:
   ```bash
   cp config.py.template config.py
   ```

### 🔹 Single-Sample Evaluation
To test a single example: `run_single_sample.py`

### 🔹 Full Evaluation
We evaluated using the batch api of openai. Therefore, we have multiple steps:
1. Generate Responses (`generate_responses.py`):
   - generate responses for a full dataset on the selected dataset / language / model.

2. Judge the Responses:
   - Create the Judges (`judge_batch.py`):
     - Update constants at the top of the script to match your target model, mode, and languages

    - Parse Judge Outputs (`parse_judge_batch.py`):
     - Update constants at the top of the script to match your target model, mode, and languages

3. Run analysis (`analysis.py`):
    - Register your model in the `MODELS` constant of the Analysis class
    - Then, invoke the desired analysis method.

## Citation

If you use any of the work, please cite the following paper:

```tex
@InProceedings{ellinger-anschtz-groh:2025:RANLP,
  author    = {Ellinger, Lukas  and  AnschÃ¼tz, Miriam  and  Groh, Georg},
  title     = {Simplifications Are Absolutists: How Simplified Language Reduces Word Sense Awareness in LLM-Generated Definitions},
  booktitle      = {Proceedings of the 15th International Conference on Recent Advances in Natural Language Processing - Natural Language Processing in the Generative AI era},
  month          = {September},
  year           = {2025},
  address        = {Varna, Bulgaria},
  publisher      = {INCOMA Ltd., Shoumen, Bulgaria},
  pages     = {342--351},
  abstract  = {Large Language Models (LLMs) can provide accurate word definitions and explanations for any context. However, the scope of the definition changes for different target groups, like children or language learners. This is especially relevant for homonymsâ€”words with multiple meaningsâ€”where oversimplification might risk information loss by omitting key senses, potentially misleading users who trust LLM outputs. We investigate how simplification impacts homonym definition quality across three target groups: Normal, Simple, and ELI5. Using two novel evaluation datasets spanning multiple languages, we test DeepSeek v3, Llama 4 Maverick, Qwen3-30B A3B, GPT-4o mini, and Llama 3.1 8B via LLM-as-Judge and human annotations. Our results show that simplification drastically degrades definition completeness by neglecting polysemy, increasing the risk of misunderstanding. Fine-tuning Llama 3.1 8B with Direct Preference Optimization substantially improves homonym response quality across all prompt types. These findings highlight the need to balance simplicity and completeness in educational NLP to ensure reliable, context-aware definitions for all learners.},
  url       = {https://aclanthology.org/2025.ranlp-1.42}
}
```
