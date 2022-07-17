import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.stats import zscore

data = pd.read_csv('WordLevelTags/finalProcessedAnnotation.csv')
data.drop(data[data['createdBy_id'] == 50].index, inplace=True)
results = defaultdict(list)

def element_wise_add(l):
    s = np.zeros(len(l[0]))
    for i in l:
        s += np.array(i)
    assert len(s) == len(l[0])
    return s

def make_gold(sentence, binary_sum, kappa_dict, binary_dict):
    tokenized = sentence.split()
    assert len(tokenized) == len(binary_sum)
    final, tags = [], []
    for i, (token, bs) in enumerate(zip(tokenized, binary_sum)):
        if bs > 2:
            final.append('[{}]'.format(token))
            tags.append('BAD')
        elif bs < 2:
            final.append(token)
            tags.append('OK')
        else:
            agreed = [k for k, v in binary_dict.items() if v[i] == 1]
            disagreed = [k for k, v in binary_dict.items() if v[i] == 0]
            print('agreed: {}'.format(agreed))
            print('disagreed: {}'.format(disagreed))
            #if kappa_dict[agreed[0]][agreed[1]] > kappa_dict[disagreed[0]][disagreed[1]]:
            if kappa_dict[agreed[0]]+kappa_dict[agreed[1]] > kappa_dict[disagreed[0]]+kappa_dict[disagreed[1]]:
                print('agreed!')
                final.append('[{}]'.format(token))
                tags.append('BAD')
            else:
                print('disagreed!')
                final.append(token)
                tags.append('OK')
    return ' '.join(final), ' '.join(tags)+' OK\n' # add one "OK" for EOS token


if __name__ == '__main__':
    '''
    kappa_dict = {
        46: {47: 0.32, 48: 0.379, 49: 0.295},
        47: {48: 0.531, 49: 0.381},
        48: {49: 0.373}
    }
    '''
    kappa_dict = {
        46: 0.332,
        47: 0.411,
        48: 0.427,
        49: 0.350
    }

    results = defaultdict(list)
    source_tag, mt_tag = '', ''
    sentence_level = defaultdict(list)
    all_scores = []
    src, mt, word_level_mt = '', '', ''
    for index, item_id in enumerate(sorted(set(data['item_id']))):
        tmp = data[data['item_id'] == item_id].sort_values(by='createdBy_id')

        results['index'].append(index)
        results['item_id'].append(item_id)
        sentence_level['index'].append(index)

        for s in ['source', 'target']:
            sentence = tmp['{}Text'.format(s)].values[0]
            binary_sequences = []
            binary_dict = {}
            for id, bs in zip(tmp['createdBy_id'], tmp['{}_binary'.format(s)]):
                bs = bs.strip('[')
                bs = bs.strip(']')
                bs = [int(i) for i in bs.split(',')]
                binary_sequences.append(bs)
                binary_dict[id] = bs
            print(index)
            print(item_id)
            print(binary_sequences)
            #binary_sum = element_wise_add(binary_sequences)
            binary_sum = [sum(x) for x in zip(*binary_sequences)]
            print(binary_sum)
            majority_label, tags = make_gold(sentence, binary_sum, kappa_dict, binary_dict)
            results[s].append(majority_label)
            if s == 'source':
                source_tag += tags
                src += sentence+'\n'
            else:
                mt_tag += tags
                mt += sentence+'\n'
                word_level_mt += sentence+' <EOS>\n'

        results['score'].append(np.mean(tmp['score']))

        original = tmp['sourceText'].values[0]
        translation = tmp['targetText'].values[0]

        sentence_level['original'].append(original)
        sentence_level['translation'].append(translation)
        sentence_level['scores'].append(list(tmp['score']))
        sentence_level['mean'].append(np.mean(tmp['score']))

    #df = pd.DataFrame.from_dict(results)
    #df.to_csv('majority_vote.csv', index=False)

    # create shared task required files

    # sentence-level scores
    z_scores = zscore(list(sentence_level['scores']))
    sentence_level['z_scores'] = list(z_scores)
    sentence_level['z_mean'] = [np.mean(z) for z in z_scores]

    sen_df = pd.DataFrame.from_dict(sentence_level)
    print(sen_df)
    sen_df.to_csv('wmt/enyo.df.short.tsv', index=False, sep='\t')


    # BAD/OK tags
    with open('wmt/src_tags.txt', 'w') as f:
        f.write(source_tag)
    with open('wmt/mt_tags.txt', 'w') as f:
        f.write(mt_tag)
    with open('wmt/word_level.tags', 'w') as f:
        f.write(mt_tag)

    # src/mt
    with open('wmt/src', 'w') as f:
        f.write(src)
    with open('wmt/mt', 'w') as f:
        f.write(mt)
    with open('wmt/word_level.mt', 'w') as f:
        f.write(word_level_mt)
