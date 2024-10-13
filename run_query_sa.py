import os
import json
from tqdm import tqdm
from query_sa import propose
from experiment_recorder import ExperimentRecorder
from llm_base.llm_structure import model_init
from utils import load_jsonl
from data_prep import get_dataset, get_persona
from dataset import HarryDataset, HarryDataset_Zero


def run_propose(
    problem,
    exp_dir: str,
    proposer_model: str = "gpt-3.5-turbo",
    proposer_template: str = "templates/gpt_entity.txt",
):
    """
      The main function for running the iterative PAS.

      Parameters
      ----------
      problem : Dataloader
        The Dataloader.
      exp_dir: str
          The directory to save the results.
      proposer_model: str
          The model to use for the proposer.          The number of descriptions to propose in each proposing round.
      proposer_template: str
      """

    # recorder = ExperimentRecorder()
    os.makedirs(exp_dir, exist_ok=True)
    # recorder.set_output_dir(exp_dir)
    labelpath = os.path.join(exp_dir, "labels.json")
    filepath = os.path.join(exp_dir, "proposed.json")
    if os.path.exists(filepath) and os.path.exists(labelpath):
        all_descriptions = load_jsonl(filepath)
        all_labels = load_jsonl(labelpath)
    else:
        # proposer
        if proposer_model == "HuggingFaceH4/zephyr-7b-beta" or proposer_model == "meta-llama/Llama-2-7b-chat-hf":
            tokenizer, proposer_model = model_init(proposer_model)
        else:
            tokenizer = None

        recorder = ExperimentRecorder()
        # os.makedirs(exp_dir, exist_ok=True)
        recorder.set_output_dir(exp_dir)
        descriptions = []
        labels = []
        all_descriptions = []
        all_labels = []
        count = 0
        for single_prob, score in tqdm(problem, desc="Proposing..."):
            text = single_prob

            new_description = propose(
                problem=text,
                proposer_model=proposer_model,
                tokenizer=tokenizer,
                proposer_template=proposer_template,
            )
            recorder.record_propose(new_description, "proposer")

            res = json.dumps(dict(content=text, label=new_description[0]))
            descriptions.append(res + "\n")
            all_descriptions.append({"content":text, "label":new_description[0]})

            res_label = json.dumps(score)
            labels.append(res_label + "\n")
            all_labels.append(score)

            if count % 200 == 0:
                with open(filepath, "a") as f:
                    f.write("".join(descriptions))
                descriptions = []

                with open(labelpath, "a") as f:
                    f.write("".join(labels))
                labels = []

            count += 1
            if count >= len(problem):
                break
        with open(filepath, "a") as f:
            f.write("".join(descriptions))
        with open(labelpath, "a") as f:
            f.write("".join(labels))
    # with open(os.path.join(exp_dir, "proposed.josn"), 'w') as f:
    #     json.dump(all_descriptions, f)

    return all_descriptions, all_labels


def gen_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",
                        type=str,
                        default="./data"
                        )
    parser.add_argument("--data_name", type=str, default="character_intersection.json")
    parser.add_argument("--aspect", type=str, default="all")
    parser.add_argument("--exp_dir", type=str, default="./experiments/")  # sentiment_analysis_culture, sentiment_analysis_religion, sentiment_analysis_subjectivity，sentiment_analysis_ideology, sentiment_analysis_vocation, sentiment_analysis_personality, sentiment_analysis_entity
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk_text_to_words", type=int, default=None)
    parser.add_argument("--turn_off_approval_before_running", action="store_true")

    parser.add_argument("--proposer_model", type=str, default="gpt-4-turbo-2024-04-09")  #gpt-4-turbo-2024-04-09 gpt-3.5-turbo-0125
    parser.add_argument(
        "--proposer_template",
        type=str,
        default="templates/gpt_sa_0.txt",
    )  # gpt_sa.txt

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = gen_args()
    model = "gpt{}".format(args.proposer_model.split('-')[1])
    if args.aspect != "0":
        args.proposer_template = "templates/gpt_sa.txt"

    else:
        args.proposer_template = "templates/gpt_sa_0.txt"
    args.exp_dir = os.path.join(args.exp_dir, model, "sentiment_analysis_{}".format(args.aspect))

    data_combined, character = get_dataset()
    character = get_persona(character, aspect=args.aspect)
    if args.aspect != "0":
        harry_data = HarryDataset(data_combined, character)
    else:
        harry_data = HarryDataset_Zero(data_combined, character)

    descriptions, labels = run_propose(problem=harry_data,
                exp_dir=args.exp_dir,
                proposer_model=args.proposer_model,
                proposer_template=args.proposer_template)

    print("done!")
