import os
import torch
import json
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl.trainer.sft_trainer import SFTTrainer, SFTConfig

# 1. Configuration & Setup
MODEL_NAME = "facebook/opt-350m"
OUTPUT_DIR = "./opt350m_event_extractor1"
DATASET_PATH = "./synthetic_event_extraction_dataset.json"

print(f"Loading tokenizer and model for: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# OPT models often lack a designated pad token; fallback to the end-of-sequence token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

# 2. Load and Format Your Synthetic JSON Dataset
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# Transform the raw validation tests into string completion formats
formatted_records = []
mock_today = "2026-05-20"
timezone = "America/New_York"

for item in raw_data:
    user_input = item.get("input", "")
    expected_output = item.get("output", {})
    
    # We construct the text to teach the base model how to autocomplete the target JSON pattern
    text_block = (
        f"Task: Extract event details as raw JSON.\n"
        f"Today's date: {mock_today}\n"
        f"Timezone: {timezone}\n"
        f"Input: \"{user_input}\"\n"
        f"Output: {json.dumps(expected_output)}{tokenizer.eos_token}\n"
    )
    formatted_records.append({"text": text_block})

# Convert to a Hugging Face Dataset object
dataset = Dataset.from_list(formatted_records)
print(f"✅ Successfully prepared {len(dataset)} examples for training.")

# 3. Define the Training Configuration
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,       
    gradient_accumulation_steps=4,       
    num_train_epochs=4,                  
    learning_rate=5e-5,                  
    logging_steps=5,
    save_strategy="epoch",
    fp16=torch.cuda.is_available(),       
    optim="adamw_torch",
    report_to="none",
    
    # Move the text field configuration here!
    dataset_text_field="text",
                      
)

# 4. Initialize SFTTrainer and Begin Training
print("🚀 Initializing training loop...")
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    processing_class=tokenizer,
    # max_seq_length=512,
    args=training_args,  # Pass the config here (it handles text field & max length)
)
trainer.train()

# 5. Save the Fine-Tuned Model and Tokenizer Local Assets
print(f"💾 Saving fine-tuned model assets to {OUTPUT_DIR}...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("🎉 Custom training process complete!")