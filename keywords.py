from ast import keyword
import json
from statistics import mean
from scipy import stats

def get_querys():
    querys = []
    with open('./data/qmsum/preprocessed/test.jsonl', 'r') as grf:
        for line in grf:
            json_line = json.loads(line)
            query = json_line['query']
            querys.append(query)
    return querys

def get_not_whole_summary_list(sum_path, querys):
    preds = []
    with open(sum_path, 'r') as rf:
        for pred, query in zip(rf, querys):
            if "whole" not in query and query != "Summarize the meeting":
                pred = pred.strip("\n").strip().lower()
                preds.append(pred)
    return preds

def recall(keywords_list, preds):
    assert len(keywords_list) == len(preds)
    overlaps = []
    for keys, pred in zip(keywords_list, preds):
        overlap_num = len([key for key in keys if key in pred])
        overlaps.append(overlap_num/len(keys))
    return overlaps

if __name__ == "__main__":
    keywords_list = [[]]
    with open('./data/qmsum/cases/keywords.txt', 'r') as rf:
        for line in rf:
            line = line.rstrip("\n").strip().lower()
            if len(line) != 0:
                keywords_list[-1].append(line)
            else:
                keywords_list.append([])
    keywords_list = [li for li in keywords_list if li != []]
    print(f'len: {len(keywords_list)}')
    
    querys = get_querys()
    preds = get_not_whole_summary_list('./output/qmsum_uscore_alpha_32_512_strided_1_0.6/selected_checkpoint/predictions.test', querys)
    uscore_overlaps = recall(keywords_list, preds)
    print(round(mean(uscore_overlaps), 4))
    # print(uscore_overlaps)

    querys = get_querys()
    preds = get_not_whole_summary_list('./output/qmsum_uscore_alpha_32_512_strided_1_0.0/selected_checkpoint/predictions.test', querys)
    baseline_overlaps = recall(keywords_list, preds)
    print(round(mean(baseline_overlaps), 4))
    # print(baseline_overlaps)

    querys = get_querys()
    preds = get_not_whole_summary_list('../ExGeSum/output/qmsum_exge_1/selected_checkpoint/predictions.test', querys)
    two_phrase_overlaps = recall(keywords_list, preds)
    print(round(mean(two_phrase_overlaps), 4))
    # print(baseline_overlaps)

    querys = get_querys()
    # preds = get_not_whole_summary_list('./data/qmsum/cases/segenc.txt', querys)
    preds = get_not_whole_summary_list('./output/qmsum_uscore_alpha_32_512_strided_3_0.0/selected_checkpoint/predictions.test', querys)
    segenc_overlaps = recall(keywords_list, preds)
    print(round(mean(segenc_overlaps), 4))
    # print(baseline_overlaps)

    querys = get_querys()
    preds = get_not_whole_summary_list('../ExGeSum/output/qmsum_raw_1/selected_checkpoint/predictions.test', querys)
    bart_overlaps = recall(keywords_list, preds)
    print(round(mean(bart_overlaps), 4))
    # print(baseline_overlaps)

    ttest = stats.ttest_ind(uscore_overlaps, baseline_overlaps)
    print(ttest)
    ttest = stats.ttest_ind(uscore_overlaps, two_phrase_overlaps)
    print(ttest)
    ttest = stats.ttest_ind(uscore_overlaps, segenc_overlaps)
    print(ttest)
    ttest = stats.ttest_ind(uscore_overlaps, bart_overlaps)
    print(ttest)
