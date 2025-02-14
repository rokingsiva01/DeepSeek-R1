from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Load model and tokenizer
model_path = "./workspaces/DeepSeek-R1/DeepSeek-R1-Distill-Qwen-1.5B"  # Path to your downloaded folder
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map="auto",  # Automatically uses GPU if available
    torch_dtype=torch.float16  # Reduces memory usage
)

# Create pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100,
    temperature=0.7  # Controls randomness (lower = more deterministic)
)

# Test inference
prompt = "Hey, how's it going? "
response = pipe(prompt)[0]['generated_text']
print(f"Response: {response}")