import json
from peft import LoraConfig
import torch
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig
from transformers.training_args import TrainingArguments  # Direct path bypasses the import check!from peft import LoraConfig  # Kept this to actually apply LoRA
from datasets import load_dataset
from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.sft_config import SFTConfig


model_id = "Qwen/Qwen3.5-0.8B" 

# 1. Quantization Configuration
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,  
    bnb_4bit_quant_type="nf4",             
    bnb_4bit_use_double_quant=True         
)

# 2. Load Base Model and Tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto"                      
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
# Qwen models don't have a pad token by default, use eos token
tokenizer.pad_token = tokenizer.eos_token 

# 3. LoRA Configuration (Crucial since you are saving a LoRA adapter later!)
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # Standard Qwen projection layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 4. Format Dataset using Qwen's native ChatML style
def format_prompts(batch):
    formatted = []
    for i, o in zip(batch['input'], batch['output']):
        # If the output dict isn't a string yet, stringify it
        if isinstance(o, dict):
            o = json.dumps(o)
        
        text = f"<|im_start|>system\nExtract JSON.<|im_end|>\n<|im_start|>user\n{i}<|im_end|>\n<|im_start|>assistant\n{o}<|im_end|>"
        formatted.append(text)
    return {"text": formatted}

dataset = load_dataset("json", data_files="test.jsonl", split="train")
dataset = dataset.map(format_prompts, batched=True)


# 1. Define SFTConfig (which holds both training args AND max_seq_length)
args = SFTConfig(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    max_steps=60,
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=10,
    output_dir="./results",
    report_to="none",
    max_length=512,  # ✨ Move max_seq_length inside SFTConfig
)

# 2. Pass the config to the trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    args=args,  
)

# 7. Train and Save
trainer.train()

# This will now successfully save just your small LoRA adapter weights!
trainer.save_model("./my_lora_adapter")
tokenizer.save_pretrained("./my_lora_adapter")

print("🎉 LoRA adapter saved to ./my_lora_adapter") 