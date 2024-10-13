import os
import json
from tqdm import tqdm
from query import propose
from experiment_recorder import ExperimentRecorder
from llm_base.llm_structure import model_init
# from utils import load_json
from utils import load_dataset


def run_propose(
        problem: object,
        exp_dir: str,
        proposer_model: str = "gpt-3.5-turbo",
        proposer_template: str = "templates/gpt_entity.txt",
) -> object:
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
    filepath = os.path.join(exp_dir, "proposed.json")
    if os.path.exists(filepath):
        all_descriptions = load_dataset(filepath)
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
        all_descriptions = []
        # all_descriptions = {}
        count = 0
        for single_prob in tqdm(problem, desc="Proposing..."):

            text = single_prob
            # for text in texts:

            new_description = propose(
                problem=text,
                proposer_model=proposer_model,
                tokenizer=tokenizer,
                proposer_template=proposer_template,
            )
            recorder.record_propose(new_description, "proposer")

            # all_descriptions.append(new_descriptions[0])
            # res = {"context":texts, "relation":new_descriptions}
            res = json.dumps(dict(name=text, label=new_description[0]))
            descriptions.append(res + "\n")
            all_descriptions.append({"name":text, "label":new_description[0]})


            if count % 200 == 0:
                with open(os.path.join(exp_dir, "proposed.json"), "a") as f:
                    f.write("".join(descriptions))
                descriptions = []

            count += 1
            # exit()
        with open(os.path.join(exp_dir, "proposed.json"), "a") as f:
            f.write("".join(descriptions))
    # with open(os.path.join(exp_dir, "proposed.josn"), 'w') as f:
    #     json.dump(all_descriptions, f)

    return all_descriptions


def gen_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",
                        type=str,
                        default="./data"
                        )
    parser.add_argument("--data_name", type=str, default="character_intersection.json")
    parser.add_argument("--aspect", type=str, default="entity")
    parser.add_argument("--exp_dir", type=str, default="./experiments/")  # entity, culture, religion, subjectivity，ideology, vocation, personality
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk_text_to_words", type=int, default=None)
    parser.add_argument("--turn_off_approval_before_running", action="store_true")

    parser.add_argument("--proposer_model", type=str, default="gpt-4-turbo-2024-04-09")  #gpt-4-turbo-2024-04-09 gpt-3.5-turbo-0125
    parser.add_argument(
        "--proposer_template",
        type=str,
        default="templates/gpt_entity.txt",
    )  # gpt_culture.txt, gpt_religion.txt, gpt_subjectivity.txt gpt_ideology.txt gpt_vocation.txt gpt_personality.txt

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = gen_args()

    args.proposer_template = "templates/gpt_{}.txt".format(args.aspect)
    args.exp_dir = os.path.join(args.exp_dir, args.aspect)

    with open(os.path.join(args.data_path, args.data_name), "r") as f:
        problem = json.load(f)

    descriptions = run_propose(problem=problem,
                exp_dir=args.exp_dir,
                proposer_model=args.proposer_model,
                proposer_template=args.proposer_template)
