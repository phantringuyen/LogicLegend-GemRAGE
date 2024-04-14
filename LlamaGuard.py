from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


model_id = "meta-llama/LlamaGuard-7b"
device = "cuda"
dtype = torch.bfloat16

LLAMA_API_KEY = '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'
login(LLAMA_API_KEY)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device)

def moderate_with_template(chat):
    input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)
    output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
    prompt_len = input_ids.shape[-1]
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

