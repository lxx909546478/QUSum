from ast import keyword
import json

keywords_list = [[]]
with open('./data/qmsum/cases/keywords.txt', 'r') as rf:
    for line in rf:
        line = line.rstrip("\n")
        if len(line) != 0:
            keywords_list[-1].append(line)
        else:
            keywords_list.append([])
keywords_list = [li for li in keywords_list if li != []]
print(keywords_list)

with open('./data/qmsum/cases/cases.txt', 'w') as wf:
    with open('./data/qmsum/preprocessed/test.jsonl', 'r') as grf, \
        open('./output/qmsum_uscore_alpha_32_512_strided_1_0.6/selected_checkpoint/predictions.test', 'r') as prf, \
            open('./output/qmsum_uscore_alpha_32_512_strided_1_0.0/selected_checkpoint/predictions.test', 'r') as rrf:
        querys = []
        targets = []
        predicts = []
        raws = []
        for line in grf:
            json_line = json.loads(line)
            query = json_line['query']
            target = json_line['target']
            querys.append(query)
            targets.append(target)
        for predict in prf:
            predict = predict.rstrip('\n')
            predicts.append(predict)
        for predict in rrf:
            predict = predict.rstrip('\n')
            raws.append(predict)
        assert len(querys) == len(predicts) == len(predicts) == len(raws)
        
        for query, target, predict, raw in zip(querys, targets, predicts, raws):
            if "whole" not in query and query != "Summarize the meeting":
                wf.write("="*50+"\n")
                wf.write(f"{query}\n\n")
                wf.write(f"{target}\n\n")
                wf.write(f"{predict}\n\n")
                wf.write(f"{raw}\n\n")
                wf.write("="*50+"\n")