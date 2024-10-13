import os
from utils import load_jsonl, parse_score, load
from data_prep import get_persona, get_dataset
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score

def load_results():
    exp_dir = './experiments/gpt3.5/'
    subpath = 'sentiment_analysis_all'
    labelpath = os.path.join(exp_dir, subpath, "labels.json")
    filepath = os.path.join(exp_dir, subpath, "proposed.json")
    all_descriptions = load_jsonl(filepath)
    all_labels = load_jsonl(labelpath)
    # all_l = load(labelpath, 'target')['target']

    all_scores = []
    for description in all_descriptions:
        label_text = description['label']
        score = parse_score(label_text,
                            character_1=description['content']['character_1'],
                            character_2=description['content']['character_2'])
        all_scores.append(score)
    return all_scores, all_labels


def collect_res_by_category(all_scores, all_labels, category_map, aspect='entity'):

    preds = {}
    targets = {}
    preds_three = {}
    targets_three = {}
    category_num = {}
    # category_senti_num = {}
    num_pos = 0
    num_neg = 0
    num_neutral = 0
    num_pred = 0
    for pred, target in zip(all_scores, all_labels):
        for name, label in target.items():
            if name != "Harry" and name != "idx":
                category_other = category_map[name][aspect]
        for name, label in target.items():
            if name != "idx":
                # if category_other not in category_senti_num:
                #     category_senti_num[category_other] = {}
                #     category_senti_num[category_other]["other"] = 0
                #     category_senti_num[category_other]["Harry"] = 0

                if label > 0:
                    num_pos += 1
                    label_three = 2  #1
                elif label == 0:
                    num_neutral += 1
                    label_three = 1  # 0
                else:
                    num_neg += 1
                    label_three = 0  #2

                if category_other not in category_num:
                    category_num[category_other] = 0
                if name != "Harry":
                    category_num[category_other] += 1

                if name in pred:
                    num_pred += 1
                    pred_score = int(pred[name])
                    if pred_score > 0:
                        pred_score_three = 2 #1
                    elif pred_score == 0:
                        pred_score_three = 1 # 0
                    else:
                        pred_score_three = 0 #2

                    # category = category_map[name][aspect]
                    if category_other not in preds.keys():
                        preds[category_other] = {'other': [], 'Harry': []}
                    if category_other not in targets.keys():
                        targets[category_other] = {'other': [], 'Harry': []}
                    if category_other not in preds_three.keys():
                        preds_three[category_other] = {'other': [], 'Harry': []}
                    if category_other not in targets_three.keys():
                        targets_three[category_other] = {'other': [], 'Harry': []}

                    if name != "Harry":
                        preds[category_other]['other'].append(pred_score)
                        targets[category_other]['other'].append(label)
                        preds_three[category_other]['other'].append(pred_score_three)
                        targets_three[category_other]['other'].append(label_three)

                    else:
                        preds[category_other]['Harry'].append(pred_score)
                        targets[category_other]['Harry'].append(label)
                        preds_three[category_other]['Harry'].append(pred_score_three)
                        targets_three[category_other]['Harry'].append(label_three)

                else:
                    pred_score = 100
                    pred_score_three = 100

                    if category_other not in preds.keys():
                        preds[category_other] = {'other': [], 'Harry': []}
                    if category_other not in targets.keys():
                        targets[category_other] = {'other': [], 'Harry': []}
                    if category_other not in preds_three.keys():
                        preds_three[category_other] = {'other': [], 'Harry': []}
                    if category_other not in targets_three.keys():
                        targets_three[category_other] = {'other': [], 'Harry': []}

                    if name != "Harry":
                        preds[category_other]['other'].append(pred_score)
                        targets[category_other]['other'].append(label)
                        preds_three[category_other]['other'].append(pred_score_three)
                        targets_three[category_other]['other'].append(label_three)

                    else:
                        preds[category_other]['Harry'].append(pred_score)
                        targets[category_other]['Harry'].append(label)
                        preds_three[category_other]['Harry'].append(pred_score_three)
                        targets_three[category_other]['Harry'].append(label_three)

    return preds, targets, preds_three, targets_three, num_pos, num_neutral, num_neg, num_pred, category_num


if __name__ == '__main__':
    _, character = get_dataset()
    character = get_persona(character, aspect="all")
    aspects = ['entity', 'culture']
    all_scores, all_labels = load_results()
    for aspect in aspects:
        (preds, targets, preds_three, targets_three,
         num_pos, num_neutral, num_neg, num_pred, category_num) = collect_res_by_category(all_scores, all_labels, character, aspect)
        success_rate = 1.0 * num_pred / (2 * len(all_labels))
        print("Aspect:", aspect, "********")
        for category in targets.keys():
            print('Category: ', category, "ccccccccc")

            print('Answer rate: ', 1.0*len(targets[category]['other'])/category_num[category])
            for group in ['other', 'Harry']:
                print('Group: ', group, "gggggggg")
                macro_f1 = f1_score(targets[category][group], preds[category][group], average='macro',labels=[-5,-4,-3,-2,-1,0,1,2,3,4,5])
                f1 = f1_score(targets[category][group], preds[category][group], average=None,labels=[-5,-4,-3,-2,-1,0,1,2,3,4,5])
                acc = accuracy_score(targets[category][group], preds[category][group])
                mse = mean_squared_error(targets[category][group], preds[category][group])

                macro_f1_three = f1_score(targets_three[category][group], preds_three[category][group], average='macro', labels=[0,1,2])
                f1_three = f1_score(targets_three[category][group], preds_three[category][group], average=None, labels=[0,1,2])
                acc_three = accuracy_score(targets_three[category][group], preds_three[category][group])
                mse_three = mean_squared_error(targets_three[category][group], preds_three[category][group])
                print("macro_f1: {}\n".format(macro_f1), "f1: {}\n".format(f1), "acc: {}\n".format(acc), "mse: {}\n".format(mse), "macro_f1_three: {}\n".format(macro_f1_three),
                      "f1_three: {}\n".format(f1_three), "acc_three: {}\n".format(acc_three), "mse_three: {}\n".format(mse_three))
        print(success_rate, num_pos, num_neg, num_neutral)