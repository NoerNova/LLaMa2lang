from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, TextStreamer
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import transformers
import torch

import chainlit as cl

# template = """
# You are a generic chatbot that always answers in Shan. Provide the answer for the following question:

# Question: {question}
# Answer:
# """

template = """
You are a generic chatbot that always answers in Shan. Provide the answer for the following question in Shan language:

Question: {question}
Answer:
"""


@cl.cache
def load_llama():
    model = "NorHsangPha/llama3_shan_finetuned"
    tokenizer = AutoTokenizer.from_pretrained(model)
    streamer = TextStreamer(tokenizer, skip_prompt=True)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
        max_length=512,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )

    llm = HuggingFacePipeline(
        pipeline=pipeline,
        model_kwargs={"temperature": 0},
    )
    return llm


llm = load_llama()


@cl.on_chat_start
async def main():
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    cl.user_session.set("llm_chain", llm_chain)

    return llm_chain


@cl.on_message
async def run(message: cl.Message):
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["Answer"]
    )

    # Retrieve the chain from the user session
    llm_chain = cl.user_session.get("llm_chain")  # type: LLMChain
    res = await llm_chain.acall(message.content, callbacks=[cb])

    if not cb.answer_reached:
        await cl.Message(content=res["text"]).send()
