from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model, TaskType
import torch

# ----------- Config -----------
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
DATASET_PATH = "./llama-v3p1-8b-instruct-dpo.jsonl"
OUTPUT_DIR = "./dpo-llama3-all"

# ----------- Load tokenizer & model -----------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer.pad_token = tokenizer.eos_token
# ----------- Apply LoRA -----------
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)

# ----------- DPO Config -----------
dpo_config = DPOConfig(
    beta=0.1,
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    num_train_epochs=10,
    logging_steps=1,
    save_strategy="no",
    report_to=[]
)

def preprocess_dpo(example):
    return {
        "prompt": example["input"]["messages"][0]["content"],
        "chosen": example["preferred_output"][0]["content"],
        "rejected": example["non_preferred_output"][0]["content"]
    }

dataset = load_dataset("json", data_files=DATASET_PATH)["train"]
dataset = dataset.map(preprocess_dpo)

# ----------- Load Trainer -----------
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,  # fallback to base model
    args=dpo_config,
    train_dataset=dataset,
    processing_class=tokenizer,
)

# ----------- Train -----------
dpo_trainer.train()

# ----------- Save -----------
dpo_trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Training complete. Model saved to:", OUTPUT_DIR)
