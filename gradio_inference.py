import bitsandbytes as bnb
from functools import partial
from peft import AutoPeftModelForCausalLM
import torch
from transformers import AutoTokenizer
import gradio as gr

# Default values
model_name = "NorHsangPha/llama3_shan_finetuned"
instruction_prompt = "သူပဵၼ်ၽူႈၸွႆႈထႅမ် ဢၼ်တေတွပ်ႇပၼ်ၶေႃႈတွပ်ႇၵူႈလွင်ႈလွင်ႈ"
thread_template_file = "threads/template_llama2.txt"
padding = "left"

# Set up device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the model and merge with base
model = AutoPeftModelForCausalLM.from_pretrained(
    model_name, device_map=device, torch_dtype=torch.bfloat16
)
model = model.merge_and_unload()
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)

if padding == "left":
    tokenizer.pad_token_id = tokenizer.eos_token_id
else:
    tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = padding

# Get the template
with open(thread_template_file, "r", encoding="utf8") as f:
    chat_template = f.read()


def chatbot(user_input, history=[]):
    thread = [{"role": "system", "content": instruction_prompt}]
    thread += [{"role": "user", "content": user_input}]

    # Prepare input in LLaMa3 chat format
    input_chat = tokenizer.apply_chat_template(
        thread, tokenize=False, chat_template=chat_template
    )
    inputs = tokenizer(
        input_chat, return_tensors="pt", padding=True, truncation=True
    ).to(device)

    # Generate response and decode
    output_sequences = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=512,
        repetition_penalty=1.2,  # LLaMa3 is sensitive to repetition
    )
    generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

    # Get the answer only
    answer = generated_text[(len(input_chat) - len(tokenizer.bos_token) + 1) :]
    thread.append({"role": "assistant", "content": answer})
    history.append((user_input, answer))

    return history, history


# Create Gradio interface
iface = gr.Interface(
    fn=chatbot,
    inputs=["text", "state"],
    outputs=["chatbot", "state"],
    title="Shan Language Chatbot",
    description="You are a generic chatbot that always answers in Shan.",
)

# Launch the interface
iface.launch()
