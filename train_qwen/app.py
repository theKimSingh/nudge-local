import torch
import json
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

app = Flask(__name__)

model_id = "Qwen/Qwen3.5-0.8B"
adapter_dir = "./my_lora_adapter"

print("🚀 Loading base model in 4-bit...")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto"
)

print("🪄 Snapping LoRA adapter on top instantly...")
model = PeftModel.from_pretrained(base_model, adapter_dir)
tokenizer = AutoTokenizer.from_pretrained(model_id)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    user_input = data.get("prompt", "")
    
    # Apply the exact ChatML formatting your LoRA expects
    prompt = f"<|im_start|>system\nExtract JSON.<|im_end|>\n<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id)
    
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Clean up the output to return just the assistant's JSON response
    response_text = decoded.split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "").strip()
    
    return jsonify({"response": response_text})

if __name__ == '__main__':
    print("🟢 API is live on http://localhost:5000/generate")
    app.run(host='0.0.0.0', port=5000)