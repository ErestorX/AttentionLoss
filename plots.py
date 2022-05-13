import matplotlib.pyplot as plt
from tabulate import tabulate
import numpy as np
import json


def compare_attDist_2models(data, model1, model2, attack):
    model1_attDist = data[model1]['AttDist_' + attack]['avgHead']
    model1_top1 = data[model1]['Metrics_' + attack]['top1']
    model2_attDist = data[model2]['AttDist_' + attack]['avgHead']
    model2_top1 = data[model2]['Metrics_' + attack]['top1']
    fig, ax = plt.subplots(figsize=(10, 5))
    blocks_id = np.arange(0, len(model1_attDist)) - (2 if 'T2T' in model1 else 0)
    for block_id, heads in zip(blocks_id, model1_attDist):
        plt.scatter(np.ones(len(heads)) * block_id, np.asarray(heads), c='r')
    blocks_id = np.arange(0, len(model2_attDist)) - (2 if 'T2T' in model2 else 0)
    for block_id, heads in zip(blocks_id, model2_attDist):
        plt.scatter(np.ones(len(heads)) * block_id, np.asarray(heads), c='b')
    plt.title('AttDist_' + attack + ' for ' + model1 + ' (Red) and ' + model2 + ' (Blue)')
    plt.savefig('plots/' + model1 + '_' + model2 + '_' + attack + '_attDist.png')


def save_attention_profile(data):
    for model in list(data.keys()):
        if 'Attention Loss' not in model:
            model_profile = data[model]['AttDist_cln']['avgHead']
            max_heads = 0
            for heads in model_profile:
                if len(heads) > max_heads:
                    max_heads = len(heads)
            for heads in model_profile:
                if len(heads) < max_heads:
                    for i in range(len(heads), max_heads):
                        heads.append(heads[0])
            with open('output/train/attentionProfile-'+model.replace('/', '-')+'.json', 'w+') as f:
                json.dump(model_profile, f)


if __name__ == '__main__':
    with open('output/val/attention_summary.json', 'r') as f:
        data = json.load(f)
    # compare_attDist_2models(data, 'T2T-ViT-t', 'Attention Loss (0.1) T2T-ViT-t', 'cln')
    # save_attention_profile(data)
