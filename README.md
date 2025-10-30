# Simplifications are Absolutists
This repository contains the code and data for our paper:

**Simplifications are Absolutists: How Simplified Language Reduces Word Sense Awareness in LLM-Generated Definitions**

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
@misc{ellinger_simplifications_2025,
	title = {Simplifications are {Absolutists}: {How} {Simplified} {Language} {Reduces} {Word} {Sense} {Awareness} in {LLM}-{Generated} {Definitions}},
	url = {http://arxiv.org/abs/2507.11981},
	author = {Ellinger, Lukas and Anschütz, Miriam and Groh, Georg},
	year={2025},
	annote = {Comment: Accepted by RANLP 2025},
}
```
