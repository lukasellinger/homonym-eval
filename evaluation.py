from tqdm import tqdm

from LLMJudge import LLMJudge
from config import Config
from llm_client import get_llm_client
from reader import JSONLineReader

PROMPT_TEMPLATES = {
    'child_en': "Explain '{word}' like I am 5 years old.",
    'child_ru': "Объясните «{word}» так, будто мне 5 лет.",
    'child_ar': "اشرح ”{word}“ وكأنني في الخامسة من عمري.",
    'child_fr': "Expliquez-moi « {word} » comme si j'avais 5 ans.",
    'child_zh': "请像我只有五岁一样解释「{word}」。",
    'child_w_context_en': "Explain '{word}' like I am 5 years old. Keep in mind that some words have more than one meaning.",
    "child_w_context_ru": "Объясните «{word}» так, как будто мне 5 лет. Пожалуйста, подумайте о различных контекстах или, если нужно, спросите.",
    "child_w_context_ar": "اشرح ”{word}“ كأنني في الخامسة من عمري. يُرجى التفكير في السياقات المختلفة أو طلب أحدها، إذا لزم الأمر.",
    "child_w_context_fr": "Expliquez-moi « {word} » comme si j'avais 5 ans. Pensez aux différents contextes ou demandez-en un, si nécessaire.",
    "child_w_context_zh": "请像我只有五岁一样解释「{word}」。如果这个词有不同的语境，请考虑这些语境，或者询问我需要哪一种。",
    'simple_en': "What is the definition of '{word}' in simple language?",
    'simple_ru': "Каково определение слова «{word}» на простом языке?",
    "simple_ar": "ما هو تعريف ”{word}“ بلغة بسيطة؟",
    "simple_fr": "Quelle est la définition de « {word} » en langage simple ?",
    "simple_zh": "用通俗的语言来说，“{word}”是什么意思？",
    'simple_w_context_en': "What is the definition of '{word}' in simple language? Keep in mind that some words have more than one meaning.",
    "simple_w_context_ru": "Что такое определение «{word}» на простом языке? Пожалуйста, подумайте о различных контекстах или, если нужно, спросите.",
    "simple_w_context_ar": "ما هو تعريف كلمة {word} بلغة بسيطة؟ يُرجى التفكير في السياقات المختلفة أو السؤال عن أحدها، إذا لزم الأمر.",
    "simple_w_context_fr": "Quelle est la définition de « {word} » en langage simple ? Pensez aux différents contextes ou demandez-en un, si nécessaire.",
    "simple_w_context_zh": "用通俗的语言来说，“{word}”是什么意思？如果这个词有不同的语境，请考虑这些语境，或者询问我需要哪一种。",
    'normal_en': "What is the definition of '{word}'?",
    'normal_ru': "Каково определение понятия «{word}»?",
    "normal_ar": "ما هو تعريف ”{word}“؟",
    "normal_fr": "Quelle est la définition du terme « {word} » ?",
    "normal_zh": "「{word}」的定义是什么？",
    'normal_w_context_en': "What is the definition of '{word}'? Keep in mind that some words have more than one meaning.",
    "normal_w_context_ru": "Каково определение «{word}»? Пожалуйста, подумайте о различных контекстах или, если нужно, спросите.",
    "normal_w_context_ar": "ما هو تعريف كلمة {word}؟ يُرجى التفكير في السياقات المختلفة أو السؤال عن أحدها، إذا لزم الأمر.",
    "normal_w_context_fr": "Quelle est la définition du terme « {word} » ? Pensez aux différents contextes ou demandez-en un, si nécessaire.",
    "normal_w_context_zh": "「{word}」的定义是什么？如果这个词有不同的语境，请考虑这些语境，或者询问我需要哪一种。",
}

class HomonymEvaluator:
    def __init__(self, response_llm, prompt_type: str):
        self.response_client = get_llm_client(response_llm)
        self.judge_client = LLMJudge()
        self.ngram_config = Config.NGRAM_CONFIG
        self.prompt_template = PROMPT_TEMPLATES[prompt_type]

    def evaluate_homonyms(self, homonyms: list[dict], output_file: str) -> list:
        results = []
        for h in tqdm(homonyms):
            word = h["word"]
            prompt = self.prompt_template.format(word=word)
            model_response = self.response_client.define_term(prompt)
            evaluation = self.judge_client.judge_response(prompt, model_response)
            result = {
                "word": word,
                "model_response": model_response,
                "evaluation": evaluation,
                "avg_google_ngrams_frequency": h["avg_google_ngrams_frequency"]
            }
            results.append(result)
            JSONLineReader().write(output_file, [result])
        return results
