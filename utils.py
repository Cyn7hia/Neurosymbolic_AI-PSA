import os
import time
import json
import re
from typing import List, Union
from copy import deepcopy
from datasets import load_dataset
import openai
from openai import OpenAI
client_emb = OpenAI(api_key="OPENAI_API_KEY")
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI(
    # This is the default and can be omitted
    # api_key=os.environ.get("OPENAI_API_KEY"),
    api_key="OPENAI_API_KEY"
)


DEFAULT_MESSAGE = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": None},
]


def parse_template(template: str) -> str:
    """
    A helper function to parse the template, which can be either a string or a path to a file.
    """
    if os.path.exists(template):
        with open(template, "r") as f:
            return f.read()
    else:
        return template


def estimate_querying_cost(
    num_prompt_toks: int, num_completion_toks: int, model: str
) -> float:
    """
    Estimate the cost of running the API, as of 2023-04-06.
    https://openai.com/pricing
    Parameters
    ----------
    num_prompt_toks : int
        The number of tokens in the prompt.
    num_completion_toks : int
        The number of tokens in the completion.
    model : str
        The model to be used.

    Returns
    -------
    float
        The estimated cost of running the API.
    """

    if model == "gpt-3.5-turbo-0125":
        cost_per_prompt_token = 0.0005 / 1000
        cost_per_completion_token = 0.0015 / 1000
    elif model == "gpt-3.5-turbo-instruct":
        cost_per_prompt_token = 0.0015 / 1000
        cost_per_completion_token = 0.002 / 1000
    elif model == "gpt-4-turbo-2024-04-09":
        cost_per_prompt_token = 0.01 / 1000
        cost_per_completion_token = 0.03 / 1000
    elif model == "gpt-4-turbo-preview":
        cost_per_prompt_token = 0.01 / 1000
        cost_per_completion_token = 0.03 / 1000
    elif model == "gpt-4":
        cost_per_prompt_token = 0.03 / 1000
        cost_per_completion_token = 0.06 / 1000
    elif model == "gpt-4-32k":
        cost_per_prompt_token = 0.06 / 1000
        cost_per_completion_token = 0.12 / 1000
    elif model.startswith("text-embedding-3-small"):
        cost_per_prompt_token = 0.00002 / 1000
        cost_per_completion_token = 0.00002 / 1000
    elif model.startswith("text-embedding-3-large"):
        cost_per_prompt_token = 0.00013 / 1000
        cost_per_completion_token = 0.00013 / 1000
    elif model.startswith("text-embedding-ada-002"):
        cost_per_prompt_token = 0.0001 / 1000
        cost_per_completion_token = 0.0001 / 1000
    elif model.startswith("davinci-002"):
        cost_per_prompt_token = 0.002 / 1000
        cost_per_completion_token = 0.002 / 1000
    else:
        raise ValueError(f"Unknown model: {model}")

    cost = (
        num_prompt_toks * cost_per_prompt_token
        + num_completion_toks * cost_per_completion_token
    )
    return cost


def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client_emb.embeddings.create(input = [text], model=model).data[0].embedding


async def chat_complete(**args):
    """
    :param args: messages=[
            {
                "role": "user",
                "content": "Say this is a test",
            }
        ],
        model="gpt-3.5-turbo",
    :return:
    """
    chat_completion = await client.chat.completions.create(
        **args
    )
    return chat_completion


class ChatGPTWrapperWithCost:
    """
    A class for openai.ChatCompletion.create() that retries when and records the cost of the API.
    """

    def __init__(self):
        self.num_queries = 0
        self.num_tokens = 0
        self.cost = 0.0

    def __call__(self, **args) -> Union[None, List[str]]:
        """
        A wrapper for openai.ChatCompletion.create() that retries 10 times if it fails.

        Parameters
        ----------
        **args
            The arguments to pass to openai.ChatCompletion.create(). This includes things like the prompt, the model, temperature, etc.

        Returns
        -------
        List[str]
            The list of responses from the API.
        """

        if args.get("messages") is None:
            args["messages"] = deepcopy(DEFAULT_MESSAGE)
            args["messages"][1]["content"] = args["prompt"]
            del args["prompt"]

        for _ in range(3):  # 10
            try:
                # responses = openai.ChatCompletion.create(**args)
                responses = asyncio.run(chat_complete(**args))
                self.num_queries += 1
                self.num_tokens += responses.usage.total_tokens
                self.cost += estimate_querying_cost(
                    responses.usage.prompt_tokens,
                    responses.usage.completion_tokens,
                    args["model"],
                )
                all_text_content_responses = [
                    c.message.content for c in responses.choices
                ]
                return all_text_content_responses
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                print(e)
                time.sleep(10)

        return None


def parse_description_response(response: str) -> List[str]:
    """
        Parse the description responses from the proposer model.

        Parameters
        ----------
        response : str
            The response from the proposer model, each description is separated by a newline, surrounded by quotes. We will extract the description within the quotes for each line.

        Returns
        -------
        List[str]
            A list of descriptions.
        """
    descriptions = []
    for line_id, line in enumerate(response.split("- ")):
        # find the two quotes
        start, end = (line.find('"') if line_id != 0 else -1), line.rfind('"')
        description = line[start + 1: end]
        if description != "":
            descriptions.append(description)

    return descriptions


def parse_label(label: str) -> str:
    """
    Parse the label from the proposer
    :param label: str
    :return:
    """
    pattern = re.compile(r'[a-zA-Z ]*')
    label = label.split("- ")[-1].strip()
    clean_label = pattern.findall(label)[1]

    return clean_label


def parse_score(response: str, character_1: str, character_2:str="Harry") -> dict:
    """
    Parse the score from the response from the proposer
    :param response:
    :return:
    """
    item_1 = '<{} to {}>'.format(character_1, character_2)
    item_2 = '<{} to {}>'.format(character_2, character_1)
    # pattern_1 = re.compile(r'<[.]+ to Harry> [+-]*[\d]+')
    # pattern_2 = re.compile(r'<Harry to [.]+> [+-]*[\d]+')
    pattern_1 = re.compile(r'{} [+-]*[\d]+'.format(item_1))
    pattern_2 = re.compile(r'{} [+-]*[\d]+'.format(item_2))
    pattern_score = re.compile(r'[+-]*[\d]+')
    texts_1 = pattern_1.findall(response)
    texts_2 = pattern_2.findall(response)
    results = {}
    if len(texts_1) > 0:
        score = pattern_score.findall(texts_1[0])
        results[character_1] = score[0]
    if len(texts_2) > 0:
        score = pattern_score.findall(texts_2[0])
        results[character_2] = score[0]

    return results



def load_json(filename):
    with open(filename, "r") as f:
        data = json.load(f)

    return data


def save_json(filename, data):
    with open(filename, "w") as f:
        json.dump(data, f)

    return

def load(path: str, name: str):
    data_files = {}
    data_files[name] = path
    extension = path.split(".")[-1]
    sents = load_dataset(
        extension, data_files=data_files, cache_dir=None
    )
    return sents


def load_jsonl(filename):
    data = []
    with open(filename, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data