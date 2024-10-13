import json
import os
import numpy as np


class ExperimentRecorder:
    def __init__(self, problem=None, label=None):
        # self.problem = problem
        # self.label = label
        # self.n = len(problem.texts)

        self.output_dir = None
        self.iteration = 0

    def record_propose(self, descriptions, name):
        with open(
            os.path.join(self.output_dir, f"iteration-{self.iteration}", f"{name}.json"),
            "w",
        ) as f:
            json.dump(
                {
                    "descriptions": descriptions,
                },
                f,
            )

    def next_iteration(self):
        self.iteration += 1
        os.makedirs(
            os.path.join(self.output_dir, f"iteration-{self.iteration}"), exist_ok=True
        )

    def set_output_dir(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(
            os.path.join(self.output_dir, f"iteration-{self.iteration}"), exist_ok=True
        )


def save_np(path, filename, data):
    np.save(os.path.join(path, filename), data)


def save_list(path, filename, data):
    with open(os.path.join(path, filename), "w") as f:
        f.writelines("\n".join(data))


def save_json(path, filename, data):
    with open(os.path.join(path, filename), "w") as f:
        json.dump(data, f)