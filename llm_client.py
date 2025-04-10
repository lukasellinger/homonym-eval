from abc import ABC, abstractmethod
import openai
from llama_cpp import Llama
from openai.types import Batch, FileObject
import requests
import os

from config import Config, LLMConfig
from reader import JSONLineReader


class LLMClient(ABC):
    def __init__(self, config: LLMConfig):
        self.model = config.model
        self.config = config

    def define_term(self, prompt: str, temperature: float = 0) -> str:
        """Generate a definition for a term."""
        return self._call_client(prompt, temperature)

    @abstractmethod
    def _call_client(self, prompt: str,temperature: float) -> str:
        """Abstract method to handle API calls for the specific provider."""
        pass


class LLamaCPPClient(LLMClient):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.llm = Llama(
            model_path=config.model,
            n_gpu_layers=-1,
            n_ctx=0,
            logits_all=True,
            n_threads=4
        )

    def _call_client(self, prompt: str, temperature: float) -> str:
        output = self.llm(
            f"<s>[INST] {prompt} [/INST]",
            max_tokens=200,
            stop=["</s>"],
            echo=False
        )
        return output['choices'][0]['text']

class OpenAIClient(LLMClient):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.client = openai.OpenAI(base_url=config.base_url, api_key=config.api_key if config.api_key else Config.CREDENTIALS.openai_api_key)

    def _call_client(self, prompt: str,temperature: float) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self._get_messages(prompt),
                temperature=temperature,
                #seed=42,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error with OpenAI {self.model}: {e}")
            return "Error: Could not process request"

    def upload_batch_file(self, file_name: str) -> FileObject:
        """
        Uploads a file to the OpenAI API for batch processing.

        :param file_name: Path to the file that needs to be uploaded.
        :return: The uploaded batch file's metadata, including the file ID.
        """
        with open(file_name, "rb") as file:
            batch_file = self.client.files.create(
                file=file,
                purpose="batch"
            )
        return batch_file

    def create_batch_job(self, file_name: str, endpoint="/v1/chat/completions") -> Batch:
        """
        Creates a batch job using the file at file_name and a specified endpoint.

        :param file_name: Path to the file to be processed in the batch job.
        :param endpoint: The API endpoint to send the batch request to. Defaults to
                         "/v1/chat/completions".
        :return: Metadata of the created batch job, including job ID.
        """
        batch_file = self.upload_batch_file(file_name)
        batch_job = self.client.batches.create(
            input_file_id=batch_file.id,
            endpoint=endpoint,
            completion_window="24h"
        )
        return batch_job

    def get_batch_result(self, output_file: str, batch_job: Batch) -> str | None:
        """
        Retrieves the result of a completed batch job and saves it to a specified file.

        :param output_file: Path where the result file should be saved.
        :param batch_job: Metadata of the batch job from which to retrieve the result.
        :return: Path to the output file where the result was saved, or None if no result is
        available.
        """
        result_file_id = batch_job.output_file_id
        if not result_file_id:
            return None

        result = self.client.files.content(result_file_id).content

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'wb') as file:
            file.write(result)

        return output_file

    @staticmethod
    def _get_messages(prompt: str):
        return [{"role": "user", "content": prompt},]

    @staticmethod
    def build_task(idx, model, content, temperature, url='/v1/chat/completion', extra_body: dict = None):
        if not extra_body:
            extra_body = {}
        return {
            "custom_id": f"task-{idx}",
            "method": "POST",
            "url": url,
            "body": {
                "model": model,
                "temperature": temperature,
                "messages": [
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                **extra_body
            }
        }

    def create_tasks(self, dataset, model, temperature, file_name, prompt_func, url='/v1/chat/completions', extra_body: dict = None):
        tasks = [self.build_task(idx, model, prompt_func(entry), temperature, url, extra_body) for idx, entry in enumerate(dataset)]
        JSONLineReader().write(file_name, tasks, mode='w')



class HuggingFaceClient(LLMClient):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.headers = {"Authorization": f"Bearer {Config.CREDENTIALS.hf_api_key}"}

    def _call_client(self, prompt: str, temperature: float) -> str:
        payload = {'messages': self._get_messages(prompt),
                   'model': self.model,
                   'parameters': {'temperature': temperature}}
        response = requests.post(self.config.api_url, headers=self.headers, json=payload, timeout=10)
        data = response.json()
        return data['choices'][0]['message']['content'].strip()

    @staticmethod
    def _get_messages(prompt: str):
        return [{"role": "user", "content": prompt}, ]



class LocalLLMClient(LLMClient):
    def __init__(self, config: LLMConfig):
        from transformers import pipeline
        super().__init__(config)
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            device_map="auto",
            torch_dtype="float16",
            token=Config.CREDENTIALS.hf_api_key,
            trust_remote_code=True,
        )

    def _call_client(self, prompt: str, temperature: float) -> str:
        output = self.pipe(
            self._get_messages(prompt),
            temperature=temperature,
        )
        return output[0]["generated_text"][1]['content'].strip()

    @staticmethod
    def _get_messages(prompt: str):
        return [{"role": "user", "content": prompt}, ]


class QwenLocalClient(LocalLLMClient):
    def __init__(self, config: LLMConfig):
        super().__init__(config)

    def _call_client(self, prompt: str, temperature: float) -> str:
        output = self.pipe(
            self._get_messages(prompt),
            temperature=0.1,
            max_new_tokens=512
        )
        return output[0]["generated_text"][1]['content'].strip()


def get_llm_client(model: str) -> LLMClient:
    config = Config.get_llm_config(model)
    if config.client_class == "OpenAIClient":
        return OpenAIClient(config)
    elif config.client_class == "LocalLLMClient":
        return LocalLLMClient(config)
    elif config.client_class == "HuggingFaceClient":
        return HuggingFaceClient(config)
    elif config.client_class == "QwenLocalClient":
        return QwenLocalClient(config)
    else:
        raise ValueError(f"Unknown client class: {config.client_class}")