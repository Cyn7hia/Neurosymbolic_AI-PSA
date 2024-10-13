from torch.utils.data import Dataset


class HarryDataset_Zero(Dataset):
    # no persona info
    def __init__(self, data, persona):
        super().__init__()
        self.data = {int(idx):value for idx, value in data.items()}
        self.persona = persona
        self.score_map = {10: 5, 9: 5, 8: 4, 7: 4, 6:3, 5:3, 4:2, 3:2, 2:1, 1:1,
                          0:0, -1:-1, -2:-1, -3:-2, -4:-2, -5:-3, -6:-3, -7:-4, -8:-4, -9:-5, -10:-5}
        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        scene = " ".join(self.data[index]['scene']) if isinstance(self.data[index]['scene'], list) else self.data[index]['scene']
        dialogue = "\n".join(self.data[index]['dialogue'])
        speakers = [name for name in self.data[index]['relations with Harry']]
        speakers.append("Harry")

        return ({"scene": scene, "dialogue_sample": dialogue, "character_1": speakers[0],
                 "character_2": speakers[1]},
                {speakers[0]: self.score_map[
                    self.data[index]['relations with Harry'][speakers[0]]["His affection to Harry"]],
                 "Harry": self.score_map[
                     self.data[index]['relations with Harry'][speakers[0]]["Harry's affection to him"]],
                 "idx": index})


class HarryDataset(Dataset):
    # ****** used for personalized sentiment analysis
    def __init__(self, data, persona):
        super().__init__()
        self.data = {int(idx):value for idx, value in data.items()}
        self.persona = persona
        self.score_map = {10: 5, 9: 5, 8: 4, 7: 4, 6:3, 5:3, 4:2, 3:2, 2:1, 1:1,
                          0:0, -1:-1, -2:-1, -3:-2, -4:-2, -5:-3, -6:-3, -7:-4, -8:-4, -9:-5, -10:-5}
        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        scene = " ".join(self.data[index]['scene']) if isinstance(self.data[index]['scene'], list) else self.data[index]['scene']
        dialogue = "\n".join(self.data[index]['dialogue'])
        speakers = [name for name in self.data[index]['relations with Harry']]
        speakers.append("Harry")

        persona_1 = ".\n".join([": ".join([attr,label]) for attr,label in self.persona[speakers[0]].items()])
        persona_2 = ".\n".join([": ".join([attr,label]) for attr,label in self.persona[speakers[1]].items()])
        return ({"scene": scene, "dialogue_sample": dialogue, "character_1": speakers[0],
                "character_2": speakers[1], "persona_1": persona_1, "persona_2": persona_2},
                {speakers[0]:self.score_map[self.data[index]['relations with Harry'][speakers[0]]["His affection to Harry"]],
                "Harry":self.score_map[self.data[index]['relations with Harry'][speakers[0]]["Harry's affection to him"]],
                 "idx": index})