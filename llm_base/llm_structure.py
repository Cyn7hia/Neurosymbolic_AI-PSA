from transformers import AutoTokenizer
import transformers
import torch

hf_token = "your_hf_token"


def model_init(model="meta-llama/Llama-2-7b-chat-hf"):
    if model != "HuggingFaceH4/zephyr-7b-beta":
        tokenizer = AutoTokenizer.from_pretrained(model)
    else:
        tokenizer = None
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    return tokenizer, pipeline


def llama_generate_response(inputs, tokenizer, pipeline):

    sequences = pipeline(
        inputs,
        # ['I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n'],
        do_sample=True,  # True,
        top_p=0.8,
        # top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        # max_length=200,
        return_full_text=False
    )
    res = []
    for seq in sequences:
        # print(f"Result: {seq['generated_text']}")
        res.append(seq['generated_text'])

    return res
# https://huggingface.co/blog/llama2#how-to-prompt-llama-2
# https://blog.futuresmart.ai/integrating-llama-2-with-hugging-face-and-langchain#heading-iii-conclusion
# https://zhuanlan.zhihu.com/p/679989153
# https://ai.meta.com/blog/5-steps-to-getting-started-with-llama-2/


def zephyr_generate_response(messages, pipeline):
    # https://huggingface.co/HuggingFaceH4/zephyr-7b-beta
    """messages = [
        {
            "role": "system",
            "content": "You are a friendly chatbot who always responds in the style of a pirate",
        },
        {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
    ]"""
    # pipeline.tokenizer.pad_token_id = pipeline.tokenizer.eos_token_id
    prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipeline(prompt, max_new_tokens=256, do_sample=True, temperature=0.8,
                                                    pad_token_id=pipeline.tokenizer.eos_token_id,
                                                    return_full_text=False) #  top_k=50, top_p=0.95,
    res = []
    for seq in outputs:
        # print(f"Result: {seq['generated_text']}")
        res.append(seq['generated_text'])
        # exit()

    return res


#https://khadkechetan.medium.com/information-extraction-with-zephyr-7b-774e773c9cb2