import pandas as pd
from sklearn.metrics import cohen_kappa_score
from scipy.stats import pearsonr, spearmanr, zscore
from scipy import stats
import numpy as np
from collections import Counter, defaultdict
from statistics import mean



df = pd.read_csv('annotations.csv')
df.drop(df[df['createdBy_id'] == 50].index, inplace = True)


source_seq = []
target_seq = []


def find_position(seq):

    c = Counter(seq)
    r = list(set(seq))

    for k, v in c.items():
        if v % 2 == 0:
            r.remove(k)

    return r




for index, row in df.iterrows():
    len_source = len(row['sourceText'].split(' '))
    len_target = len(row['targetText'].split(' '))

    source = np.zeros(len_source)
    target = np.zeros(len_target)


    if type(row['reference_errors'])==str:
        source_err = row['reference_errors'].split('|')
        pos_source = find_position(source_err)
        for p in pos_source:
            source[int(p)] = 1


    if type(row['translation_errors'])==str:
        target_err = row['translation_errors'].split('|')
        pos_target = find_position(target_err)
        for p in pos_target:
            target[int(p)] = 1

    source_seq.append(source)
    target_seq.append(target)
df['source'] = source_seq
df['target'] = target_seq

df.to_csv('annotations_processed.csv', index=False)




annotators = sorted(list(set(df['createdBy_id'])))
print(annotators)

kappa_source, kappa_target, kappa_ave, pearson, spearman = [],[],[],[],[]
annotators_1, annotators_2 = [], []
numbers = []


for i in range(len(annotators)):
    for j in range(i+1, len(annotators)):
        a1 = annotators[i]
        a2 = annotators[j]


        tmp1 = df[df['createdBy_id']==a1]
        tmp2 = df[df['createdBy_id']==a2]
        item_ids = set(tmp1['item_id']).intersection(set(tmp2['item_id']))
        tmp1 = tmp1[tmp1['item_id'].isin(item_ids)].sort_values(by='item_id')
        tmp2 = tmp2[tmp2['item_id'].isin(item_ids)].sort_values(by='item_id')

        assert tmp1['item_id'].tolist() == tmp2['item_id'].tolist()

        source_tmp, target_tmp = [], []
        for v1,v2 in zip(tmp1['source'].tolist(), tmp2['source'].tolist()):
            if (v1 == v2).all():
                kappa = 1
            else:
                kappa = cohen_kappa_score(v1, v2)
            source_tmp.append(kappa)

        for v1, v2 in zip(tmp1['target'].tolist(), tmp2['target'].tolist()):
            if (v1 == v2).all():
                kappa = 1
            else:
                kappa = cohen_kappa_score(v1, v2)
            target_tmp.append(kappa)
        annotators_1.append(a1)
        annotators_2.append(a2)

        kappa_source.append(np.average(source_tmp))
        kappa_target.append(np.average(target_tmp))
        kappa_ave.append(np.average([np.average(source_tmp), np.average(target_tmp)]))

        pearson.append(pearsonr(tmp1['score'],tmp2['score'])[0])
        spearman.append(spearmanr(tmp1['score'],tmp2['score'])[0])

        numbers.append(len(item_ids))

results = pd.DataFrame()
results['annotator_1'] = annotators_1
results['annotator_2'] = annotators_2
results['number_items'] = numbers
results['kappa_source'] = kappa_source
results['kappa_target'] = kappa_target
results['kappa_ave'] = kappa_ave
results['pearson'] = pearson
results['spearman'] = spearman


print(results)
results.to_csv('annotation1010.csv',index=False)
