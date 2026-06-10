import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_id = "Qwen/Qwen3.5-0.8B"
adapter_dir = "./my_lora_adapter"
output_dir = "./qwen3.5-json-merged"

print("⏳ Loading base model and adapter...")
# Load the base model in float16 (do not use 4-bit quantization here!)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="cpu" # Use CPU to avoid running out of VRAM during merging
)

# Load the adapter on top of the base model
model = PeftModel.from_pretrained(base_model, adapter_dir)

print("🔄 Merging weights...")
# Permanently fuse the LoRA layers into the base architecture
model = model.merge_and_unload()

print(f"💾 Saving merged model to {output_dir}...")
# Save the complete model and tokenizer
model.save_pretrained(output_dir)
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.save_pretrained(output_dir)

print("🎉 Done! Base model and LoRA are successfully fused.")