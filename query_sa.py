from utils import parse_template
import utils
from typing import List
from dataclasses import dataclass
from llm_base.llm_structure import llama_generate_response, zephyr_generate_response
from query import ProposerResponse


def construct_proposer_prompt(
    problem: dict,
    template: str,
):
    """
    Construct the prompt for the proposer model.

    Parameters
    ----------
    problem : dict
        Text problem to be included in the prompt.
    template : str
        The template used for proposing, can be the actual string, or path to template file

    Returns
    -------
    str
        The formatted prompt for the proposer model.
    """

    prompt = parse_template(template).format(
        **problem
    )
    return prompt


def propose_description(
    problem: str|dict,
    model, #: str,
    tokenizer,
    template: str,
    # random_seed: int,
    log_propose_prompt=False,
) -> ProposerResponse:
    """
    Propose descriptions for a given problem.

    Parameters
    ----------
    problem : PropData
        The PropData instance.
    text_subset : str
        The text samples to be included in the prompt. T in the paper.
    relations: List[str]
        The relation type list containing all the hypothesized relation types in the previous step.
    model : str
        The model to use for proposing descriptions.
    template : str
        The template used for proposing, can be the actual string, or path to template file
    example_descriptions : List[str]
        A list of example descriptions provided for formatting reference.
    num_descriptions_per_round : int
        The number of descriptions the model should suggest. J' in the paper.
    random_seed : int
        The random seed for sampling text samples.
    log_propose_prompt : bool
        Whether to log the prompt used for proposing.

    Returns
    -------
    ClusterProposerResponse
        The response from the proposer model. This includes the descriptions, the prompt, and the text samples used in the prompt.
    """
    # set the random seed
    # random.seed(random_seed)

    # construct the prompt based on the text samples and the goal
    proposer_prompt = construct_proposer_prompt(
        problem,
        template,
    )

    # get the response from the model
    if log_propose_prompt:
        print("Running the proposer model...")
        print(f"{proposer_prompt}")
    if isinstance(model, str):
        chat_gpt_query_model = utils.ChatGPTWrapperWithCost()
        raw_response = chat_gpt_query_model(
            prompt=proposer_prompt, model=model, temperature=0.7  # 0.2
        )

    elif "Mistral" in model.model.__class__.__name__:  # model == "HuggingFaceH4/zephyr-7b-beta":
        guide_deep_sys = "You are an expert in NLP, who helps to cluster the sentences based on relation types."
        messages=[{"role": 'system', "content": guide_deep_sys}, {"role": 'user', "content": proposer_prompt}]

        raw_response = zephyr_generate_response(messages=messages, pipeline=model)

    elif "Llama" in model.model.__class__.__name__:  # model == "meta-llama/Llama-2-7b-chat-hf":
        raw_response = llama_generate_response(proposer_prompt, tokenizer=tokenizer, pipeline=model)

    if log_propose_prompt:
        print("Proposer model response:")
        print(raw_response)
    if raw_response is None:
        return ProposerResponse(
            description="",
            proposer_prompt=proposer_prompt,
            problem=problem,
            raw_response="",
        )
    text_response = raw_response[0]

    # parse the response to get the descriptions
    # each description is separated by a newline, surrounded by quotes according to the prompt
    description = utils.parse_description_response(text_response)
    # description = utils.parse_label(description[0])
    # description = utils.parse_proposed_relations(description)

    # the later ones could very likely be of lower quality.
    # description = description[0]
    description = ", ".join(description)
    parse_scores = utils.parse_score(description, character_1=problem['character_1'], character_2=problem['character_2'])
    if len(parse_scores) < 2:
        return ProposerResponse(
            description="",
            proposer_prompt=proposer_prompt,
            problem=problem,
            raw_response="",
        )

    # return the descriptions, the prompt, and the text samples used in the prompt
    return ProposerResponse(
        description=description,
        proposer_prompt=proposer_prompt,
        problem=problem,
        raw_response=text_response,
    )


def propose(
    problem: List[str]|str|dict,
    proposer_model, #: str = "gpt-3.5-turbo",
    tokenizer,
    proposer_template: str = "templates/gpt_proposer_short_0.txt",
    time_thresh: int=1,#3,

) -> List[str]:
    """
    Proposal stage in the paper, which result in a list of candidate explanations for clusters. mainly calls the propose_descriptions_multi_round function.

    Parameters
    ----------
    goal: str,
    problem : List[str],
        The list of sentences (default 1 sentence).
    example_descriptions : List[str]
        The example descriptions to use in the prompt. used to clarify what the goal is using some example descriptions.
    proposer_model : str, optional
        The model used to propose descriptions, by default "gpt-3.5-turbo"
    proposer_num_descriptions_per_round : int, optional
        The number of descriptions to propose per round, by default 8
    proposer_template : str, optional
        The template used to construct the prompt, by default "templates/gpt_cluster_proposer_short.txt"; can switch to proposing more detailed descriptions by using "templates/gpt_cluster_proposer_detailed.txt"

    Returns
    -------
    List[str]
        The proposed descriptions. (we use descriptions and explanations interchangeably)
    """
    # load the hypothesized relation types
    # relations = load_json(relation_path)

    descriptions = []
    obtained_res = False
    repeat_time = 0
    if isinstance(problem, list):
        for data in problem:
            # text = data["context"]
            # obtain the proposer results for multiple rounds
            proposer_results = propose_description(
                problem=data,
                model=proposer_model,
                tokenizer=tokenizer,
                template=proposer_template,
            )

            # gather the descriptions for each sample
            descriptions.append(proposer_results.description)
    else:
        while not obtained_res and repeat_time < time_thresh:
            proposer_results = propose_description(
                problem=problem,
                model=proposer_model,
                tokenizer=tokenizer,
                template=proposer_template,
            )
            if proposer_results.description != 0:
                obtained_res = True

        # gather the descriptions for each sample
        descriptions.append(proposer_results.description)

    return descriptions