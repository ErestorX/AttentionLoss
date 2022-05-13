import matplotlib.pyplot as plt
from tabulate import tabulate
import numpy as np
import json


def beautiful_model_name(model_name):
    final_tokens = []
    for token in model_name.split("_"):
        if token == 't2t':
            final_tokens.append('T2T-')
        elif token == 'vit':
            final_tokens.append('ViT')
        elif token == 'base':
            final_tokens.append('-B')
        elif token == 'small':
            final_tokens.append('-S')
        elif token == 'tiny':
            final_tokens.append('-T')
        elif token == '14':
            pass
        elif token == 'patch16':
            final_tokens.append('/16')
        elif token == 'patch32':
            final_tokens.append('/32')
        elif token == '224':
            pass
        elif token == 'p':
            final_tokens.append('-p')
        elif token == 't':
            final_tokens.append('-t')
        elif token == 'pretrained':
            final_tokens.append(' Pretrained')
        elif token == 'scratch':
            final_tokens.append(' INet training')
        elif token == 'doexp05l':
            final_tokens.append(' - dropout: 0.5/exp(0.5*l*d)')
        elif token == 'donegexp05l':
            final_tokens.append(' - dropout: 0.5-0.5/exp(0.5*l*d)')
        elif token == 'donegexp025l':
            final_tokens.append(' - dropout: 0.5-0.5/exp(0.25*l*d)')
        elif token == 'donegexp075l':
            final_tokens.append(' - dropout: 0.5-0.5/exp(0.75*l*d)')
        elif token == 'doexp5':
            final_tokens.append(' - dropout: 0.5/exp(5*d)')
        elif token == 'finetuned':
            final_tokens.append(' for INet fine tuning')
        else:
            final_tokens.append('-'+token)
    return ''.join(final_tokens)


def beautiful_attack_name(attack_name, cln_is_no=False):
    if attack_name == '_cln':
        return 'no' if cln_is_no else 'Clean'
    else:
        param = attack_name.split('_')[-2:]
        param = [float(p.split(':')[-1]) for p in param]
        steps, eps = param[0], param[1]
        atk_name = ('FGSM' if steps == 1 else 'PGD') + ' ' + str(eps)
        return atk_name


def get_top1_val_base(data, attacks, model_list):
    t2t_blue = ['midnightblue', 'mediumblue', 'blue', 'mediumslateblue', 'darkorchid', 'fuchsia', 'violet', 'hotpink']
    t2t_blue.reverse()
    blue_id = 0
    vit_green = ['darkgreen', 'green', 'limegreen', 'mediumseagreen', 'aquamarine', 'turquoise', 'paleturquoise',
                 'lightseagreen', 'darkcyan']
    green_id = 0
    if model_list is None:
        model_list = list(data.keys())
    cln = np.asarray([data[model]['Metrics_cln']['top1'] for model in model_list])
    exp_title = [beautiful_attack_name('_cln')]
    per_model_evol = np.reshape(cln, (cln.shape[0], 1))
    for atk in attacks:
        exp_title.append(beautiful_attack_name(atk))
        experiment = 'Metrics_adv' + atk
        adv = np.asarray([data[model][experiment]['top1'] for model in model_list])
        adv = adv.reshape((adv.shape[0], 1))
        per_model_evol = np.concatenate((per_model_evol, adv), axis=1)
    plt.figure(figsize=(12, 5))
    for i in range(per_model_evol.shape[0]):
        if 't2t' in model_list[i]:
            color = t2t_blue[blue_id]
            blue_id += 1
        else:
            color = vit_green[green_id]
            green_id += 1
        plt.plot(exp_title, per_model_evol[i], label=beautiful_model_name(model_list[i]), color=color)

    plt.axvspan(exp_title[1], exp_title[4], facecolor='grey', alpha=0.3)
    plt.legend()
    plt.grid()
    plt.ylabel('Top-1 accuracy')
    plt.title('Top-1 accuracy on clean and adversarial data')
    plt.tight_layout()
    plt.savefig('Plots/Clean_vs_Adversarial_acc_baseModels.png')
    plt.close()


def get_top1_val_all(data, attacks, model_list):
    t2t_blue = ['midnightblue', 'mediumblue', 'blue', 'mediumslateblue', 'darkorchid', 'fuchsia', 'violet', 'hotpink']
    t2t_blue.reverse()
    blue_id = 0
    vit_green = ['darkgreen', 'green', 'limegreen', 'mediumseagreen', 'aquamarine', 'turquoise', 'paleturquoise',
                 'lightseagreen', 'darkcyan']
    green_id = 0
    if model_list is None:
        model_list = list(data.keys())
    cln = np.asarray([data[model]['Metrics_cln']['top1'] for model in model_list])
    exp_title = [beautiful_attack_name('_cln')]
    per_model_evol = np.reshape(cln, (cln.shape[0], 1))
    for atk in attacks:
        exp_title.append(beautiful_attack_name(atk))
        experiment = 'Metrics_adv' + atk
        adv = np.asarray([data[model][experiment]['top1'] for model in model_list])
        adv = adv.reshape((adv.shape[0], 1))
        per_model_evol = np.concatenate((per_model_evol, adv), axis=1)
    plt.figure(figsize=(12, 5))
    for i in range(per_model_evol.shape[0]):
        if 't2t' in model_list[i]:
            color = t2t_blue[blue_id]
            blue_id += 1
        else:
            color = vit_green[green_id]
            green_id += 1
        plt.plot(exp_title, per_model_evol[i], label=beautiful_model_name(model_list[i]), color=color)

    plt.axvspan(exp_title[1], exp_title[4], facecolor='grey', alpha=0.3)
    plt.legend()
    plt.grid()
    plt.ylabel('Top-1 accuracy')
    plt.title('Top-1 accuracy on clean and adversarial data')
    plt.tight_layout()
    plt.savefig('Plots/Clean_vs_Adversarial_acc_allModels.png')
    plt.close()


def compare_att_distances_attack_avg_baseModels(data, attack, models):
    t2t_blue = ['midnightblue', 'mediumblue', 'blue', 'mediumslateblue', 'darkorchid', 'fuchsia', 'violet', 'hotpink']
    t2t_blue.reverse()
    blue_id = 0
    vit_green = ['darkgreen', 'green', 'limegreen', 'mediumseagreen', 'aquamarine', 'turquoise', 'paleturquoise',
                 'lightseagreen', 'darkcyan']
    green_id = 0
    colors = []
    list_exp =[]
    legends = []
    list_top1 = []
    max_top1 = -1
    for model in models:
        if attack in data[model]:
            list_exp.append(data[model][attack])
            list_top1.append(data[model]['Metrics_'+'_'.join(attack.split('_')[1:])]['top1'])
            if data[model]['Metrics_'+'_'.join(attack.split('_')[1:])]['top1'] > max_top1:
                max_top1 = data[model]['Metrics_'+'_'.join(attack.split('_')[1:])]['top1']
            legends.append(beautiful_model_name(model) + '\nTop1: ')
    for id, (top1, legend) in enumerate(zip(list_top1, legends)):
        if top1 == max_top1:
            legends[id] = legend + "{:.2f}   -- Best --".format(top1)
        else:
            legends[id] = legend + "{:.2f}".format(top1)
    nb_exp = len(list_exp)
    colors = colors[:nb_exp]
    list_blocks = []
    t2t = False
    for i in range(nb_exp):
        if 't2t' in models[i]:
            t2t = True
            colors.append(t2t_blue[blue_id])
            blue_id += 1
            list_blocks.append(np.arange(-2, len(list_exp[i]) - 2, 1.0))
        else:
            colors.append(vit_green[green_id])
            green_id += 1
            list_blocks.append(np.arange(0, len(list_exp[i]), 1.0))
    fig, ax = plt.subplots(figsize=(10, 5))
    if attack == 'AttDist_cln':
        title = 'Clean data'
    else:
        title = beautiful_attack_name(attack)
    plt.title('Attention distance on ' + title)
    for id, e in enumerate(list_exp):
        avg_e = []
        for block in e:
            avg_e.append(np.mean(block))
        list_exp[id] = avg_e
    for blocks, e, color, legend in zip(list_blocks, list_exp, colors, legends):
        plt.plot(blocks, e, color=color, label=legend)
    ax.yaxis.grid(True)
    ax.set_ylabel('Attention distance, in pixels')
    ax.set_xlabel('Block ID, negative for the T2T blocks')
    plt.tight_layout()
    ax.legend()
    if t2t:
        plt.axvline(x=-.5, color='grey', alpha=.8)
    plt.savefig('Plots/AttDist_' + title + '_avg_baseModels.png')
    plt.close()


def compare_att_distances_attack_avg_vitModels(data, attack, models):
    t2t_blue = ['midnightblue', 'mediumblue', 'blue', 'mediumslateblue', 'darkorchid', 'fuchsia', 'violet', 'hotpink']
    t2t_blue.reverse()
    blue_id = 0
    vit_green = ['darkgreen', 'green', 'limegreen', 'mediumseagreen', 'aquamarine', 'turquoise', 'paleturquoise',
                 'lightseagreen', 'darkcyan']
    green_id = 0
    colors = []
    list_exp =[]
    legends = []
    list_top1 = []
    max_top1 = -1
    for model in models:
        if attack in data[model]:
            list_exp.append(data[model][attack])
            list_top1.append(data[model]['Metrics_'+'_'.join(attack.split('_')[1:])]['top1'])
            if data[model]['Metrics_'+'_'.join(attack.split('_')[1:])]['top1'] > max_top1:
                max_top1 = data[model]['Metrics_'+'_'.join(attack.split('_')[1:])]['top1']
            legends.append(beautiful_model_name(model) + '\nTop1: ')
    for id, (top1, legend) in enumerate(zip(list_top1, legends)):
        if top1 == max_top1:
            legends[id] = legend + "{:.2f}   -- Best --".format(top1)
        else:
            legends[id] = legend + "{:.2f}".format(top1)
    nb_exp = len(list_exp)
    colors = colors[:nb_exp]
    list_blocks = []
    t2t = False
    for i in range(nb_exp):
        if 't2t' in models[i]:
            t2t = True
            colors.append(t2t_blue[blue_id])
            blue_id += 1
            list_blocks.append(np.arange(-2, len(list_exp[i]) - 2, 1.0))
        else:
            colors.append(vit_green[green_id])
            green_id += 1
            list_blocks.append(np.arange(0, len(list_exp[i]), 1.0))
    fig, ax = plt.subplots(figsize=(12, 5))
    if attack == 'AttDist_cln':
        title = 'Clean data'
    else:
        title = beautiful_attack_name(attack)
    plt.title('Attention distance on ' + title)
    for id, e in enumerate(list_exp):
        avg_e = []
        for block in e:
            avg_e.append(np.mean(block))
        list_exp[id] = avg_e
    colors_vit = ['darkgreen', 'green', 'aquamarine', 'turquoise', 'lightseagreen', 'darkcyan']
    colors = colors_vit
    for blocks, e, color, legend in zip(list_blocks, list_exp, colors, legends):
        plt.plot(blocks, e, color=color, label=legend)
    ax.yaxis.grid(True)
    ax.set_ylabel('Attention distance, in pixels')
    ax.set_xlabel('Block ID, negative for the T2T blocks')
    plt.tight_layout()
    ax.legend()
    if t2t:
        plt.axvline(x=-.5, color='grey', alpha=.8)
    plt.savefig('Plots/AttDist_' + title + '_avg_vitModels.png')
    plt.close()


def compare_att_distances_attack_avg_t2tModels(data, attack, models):
    t2t_blue = ['midnightblue', 'mediumblue', 'blue', 'mediumslateblue', 'darkorchid', 'fuchsia', 'violet', 'hotpink']
    t2t_blue.reverse()
    blue_id = 0
    vit_green = ['darkgreen', 'green', 'limegreen', 'mediumseagreen', 'aquamarine', 'turquoise', 'paleturquoise',
                 'lightseagreen', 'darkcyan']
    green_id = 0
    colors = []
    list_exp =[]
    legends = []
    list_top1 = []
    max_top1 = -1
    for model in models:
        if attack in data[model]:
            list_exp.append(data[model][attack])
            list_top1.append(data[model]['Metrics_'+'_'.join(attack.split('_')[1:])]['top1'])
            if data[model]['Metrics_'+'_'.join(attack.split('_')[1:])]['top1'] > max_top1:
                max_top1 = data[model]['Metrics_'+'_'.join(attack.split('_')[1:])]['top1']
            legends.append(beautiful_model_name(model) + '\nTop1: ')
    for id, (top1, legend) in enumerate(zip(list_top1, legends)):
        if top1 == max_top1:
            legends[id] = legend + "{:.2f}   -- Best --".format(top1)
        else:
            legends[id] = legend + "{:.2f}".format(top1)
    nb_exp = len(list_exp)
    colors = colors[:nb_exp]
    list_blocks = []
    t2t = False
    for i in range(nb_exp):
        if 't2t' in models[i]:
            t2t = True
            colors.append(t2t_blue[blue_id])
            blue_id += 1
            list_blocks.append(np.arange(-2, len(list_exp[i]) - 2, 1.0))
        else:
            colors.append(vit_green[green_id])
            green_id += 1
            list_blocks.append(np.arange(0, len(list_exp[i]), 1.0))
    fig, ax = plt.subplots(figsize=(12, 5))
    if attack == 'AttDist_cln':
        title = 'Clean data'
    else:
        title = beautiful_attack_name(attack)
    plt.title('Attention distance on ' + title)
    for id, e in enumerate(list_exp):
        avg_e = []
        for block in e:
            avg_e.append(np.mean(block))
        list_exp[id] = avg_e
    colors_t2t = ['hotpink', 'violet', 'darkorchid', 'blue', 'mediumblue', 'midnightblue']
    colors = colors_t2t
    for blocks, e, color, legend in zip(list_blocks, list_exp, colors, legends):
        plt.plot(blocks, e, color=color, label=legend)
    ax.yaxis.grid(True)
    ax.set_ylabel('Attention distance, in pixels')
    ax.set_xlabel('Block ID, negative for the T2T blocks')
    plt.tight_layout()
    ax.legend()
    if t2t:
        plt.axvline(x=-.5, color='grey', alpha=.8)
    plt.savefig('Plots/AttDist_' + title + '_avg_t2tModels.png')
    plt.close()


def compare_att_distances_model_avg(data, model, attacks):
    FGSM_blue = ['midnightblue', 'mediumblue', 'blue', 'mediumslateblue', 'darkorchid', 'fuchsia', 'violet', 'hotpink']
    FGSM_blue.reverse()
    blue_id = 0
    PGD_green = ['darkgreen', 'green', 'limegreen', 'mediumseagreen', 'aqquamarine', 'turquoise', 'paleturquoise',
                 'lightseagreen', 'darkcyan']
    green_id = 0
    colors = ['black']
    list_exp = [data[model]['AttDist_cln']]
    legends = ['Clean']
    for attack in attacks:
        if 'AttDist_adv' + attack in data[model]:
            list_exp.append(data[model]['AttDist_adv' + attack])
            param = attack.split('_')[1:]
            param = [float(p.split(':')[-1]) for p in param]
            steps, eps = param[0], param[1]
            if steps == 1:
                colors.append(FGSM_blue[blue_id])
                blue_id += 1
            else:
                colors.append(PGD_green[green_id])
                green_id += 1
            title = ('FGSM' if steps == 1 else 'PGD') + ' ' + str(eps)
            legends.append(title)
    nb_exp = len(list_exp)
    list_blocks = []
    for i in range(nb_exp):
        if 't2t' in model:
            list_blocks.append(np.arange(-2, len(list_exp[i]) - 2, 1.0))
        else:
            list_blocks.append(np.arange(0, len(list_exp[i]), 1.0))
    fig, ax = plt.subplots(figsize=(12, 5))
    plt.title('Attention distance on ' + beautiful_model_name(model))
    for id, e in enumerate(list_exp):
        avg_e = []
        for block in e:
            avg_e.append(np.mean(block))
        list_exp[id] = avg_e
    for blocks, e, color, legend in zip(list_blocks, list_exp, colors, legends):
        plt.plot(blocks, e, color=color, label=legend)
    ax.yaxis.grid(True)
    ax.set_xlabel('Block ID, negative for the T2T blocks')
    ax.set_ylabel('Attention distance, in pixels')
    if 't2t' in model:
        plt.axvline(x=-.5, color='grey', alpha=.8)
    plt.legend()
    plt.tight_layout()
    plt.savefig('Plots/AttDist_' + model + '_avg.png')
    plt.close()


def CKA_and_attDist_plot(data, model_1, model_2, attack):
    pad = False
    if 't2t' in model_1 and 't2t' not in model_2 or 't2t' in model_2 and 't2t' not in model_1:
        pad = True
    if attack == '_cln':
        key = 'CKA_cln'
    else:
        key = 'CKA_trf' + attack
    try:
        CKA_model_1 = np.asarray(data[model_1][key][model_1])
        CKA_model_2 = np.asarray(data[model_2][key][model_2])
        if len(CKA_model_1) > 16:
            CKA_model_1 = CKA_model_1[3::3, 3::3]
        CKA_model_1 = (CKA_model_1 - np.min(CKA_model_1)) / (np.max(CKA_model_1) - np.min(CKA_model_1))
        if len(CKA_model_2) > 16:
            CKA_model_2 = CKA_model_2[3::3, 3::3]
        CKA_model_2 = (CKA_model_2 - np.min(CKA_model_2)) / (np.max(CKA_model_2) - np.min(CKA_model_2))
    except:
        print('CKA not available for ' + model_1 + ' or ' + model_2)
        return
    if attack == '_cln':
        key = 'AttDist_cln'
    else:
        key = 'AttDist_adv' + attack
    try:
        attDist_model1 = np.asarray(data[model_1][key])
        attDist_model2 = np.asarray(data[model_2][key])
    except:
        print('AttDist' + attack + ' not available for ' + model_1 + ' or ' + model_2)
        return

    if 't2t' not in model_1 and pad:
        CKA_model_1 = np.pad(CKA_model_1, ((2, 2), (2, 2)), 'constant', constant_values=0)
    if 't2t' not in model_2 and pad:
        CKA_model_2 = np.pad(CKA_model_2, ((2, 2), (2, 2)), 'constant', constant_values=0)
    avg = []
    for block in attDist_model1:
        avg.append(np.mean(block))
    attDist_model1 = np.asarray(avg)
    avg = []
    for block in attDist_model2:
        avg.append(np.mean(block))
    attDist_model2 = np.asarray(avg)
    pad_1 = 2 if 't2t' not in model_1 and pad else 0
    pad_2 = 2 if 't2t' not in model_2 and pad else 0
    blocks_model1 = list(range(pad_1, len(attDist_model1) + pad_1, 1))
    blocks_model2 = list(range(pad_2, len(attDist_model2) + pad_2, 1))
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 5))
    im = axes[0].imshow(CKA_model_1, cmap='Reds')
    axes[0].set_title(beautiful_model_name(model_1) + '\nCKA')
    axes[0].set_xticks(range(0, len(CKA_model_1), 2))
    axes[0].set_yticks(range(0, len(CKA_model_1), 2))
    axes[0].set_xlabel('Blocks ID')
    axes[0].set_ylabel('Blocks ID')
    plt.colorbar(im, ax=axes[0])
    im = axes[2].imshow(CKA_model_2, cmap='Blues')
    axes[2].set_title(beautiful_model_name(model_2) + '\nCKA')
    axes[2].set_xticks(range(0, len(CKA_model_2), 2))
    axes[2].set_yticks(range(0, len(CKA_model_2), 2))
    axes[2].set_xlabel('Blocks ID')
    axes[2].set_ylabel('Blocks ID')
    plt.colorbar(im, ax=axes[2])
    axes[1].plot(attDist_model1, blocks_model1, color='red', label='Left model')
    axes[1].plot(attDist_model2, blocks_model2, color='blue', label='Right model')
    axes[1].set_ylim(axes[1].get_ylim()[::-1])
    axes[1].set_xlabel('Attention distance, in pixels')
    axes[1].set_ylabel('Blocks ID')
    min_x = min(int(np.min(attDist_model1)) - int(np.min(attDist_model1) % 10), int(np.min(attDist_model2)) - int(np.min(attDist_model2) % 10))
    max_x = max(int(np.max(attDist_model1)) + int(np.max(attDist_model1) % 10), int(np.max(attDist_model2)) + int(np.max(attDist_model2) % 10))
    axes[1].set_xlim([min_x, max_x])
    axes[1].set_xticks(range(min_x, max_x, 10))
    axes[1].set_yticks(range(0, int(np.max(blocks_model1)), 2))

    if attack == '_cln':
        attack_name = 'no'
    else:
        param = attack.split('_')[-2:]
        param = [float(p.split(':')[-1]) for p in param]
        steps, eps = param[0], param[1]
        attack_name = ('FGSM' if steps == 1 else 'PGD') + ' ' + str(eps)
    axes[1].set_title('Attention distance under ' + attack_name + ' attack')
    axes[1].legend()
    axes[1].set_aspect(1.0/axes[1].get_data_ratio(), adjustable='box')
    plt.tight_layout()
    plt.savefig('Plots/' + model_1 + '_' + model_2 + attack + '.png')
    plt.close()


def plot_do_fn():
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 5))
    colors = ['b', 'r', 'g', 'c']
    parameters = [1, 5, 10, 14]
    parameters_name = ["Layer: l = "+str(i) for i in parameters]
    x = np.linspace(0, 1, 100)
    for l, c in zip(parameters, colors):
        d1 = 0.5 - (0.5 / np.exp(l * x / 2))
        d2 = 0.5 / np.exp(l * x / 2)
        axes.plot(d1, '--', color=c)
        axes.plot(d2, '-', color=c)
    lines = axes.get_lines()
    legend1 = plt.legend([lines[0], lines[1]], ["Drop out = 0.5 - (0.5 / exp(lpx))", "Drop out = 0.5 / exp(lpx)"], loc=3)
    legend2 = plt.legend([lines[i] for i in [0, 2, 4, 6]], parameters_name, loc=2)
    axes.add_artist(legend1)
    axes.add_artist(legend2)
    axes.set_xlabel('Attention distance in percent of feature space')
    axes.set_ylabel('Attributed dropout')
    plt.grid()
    plt.title('Drop out functions, with function parameter p = 0.5')
    plt.tight_layout()
    plt.savefig('Plots/do_fn.png')
    plt.close()


def plot_CKA_clean(data, list_model_1, list_model_2):
    fig, axes = plt.subplots(nrows=1, ncols=len(list_model_1), figsize=(12, 5))
    for id, (model_1, model_2) in enumerate(zip(list_model_1, list_model_2)):
        mat = np.asarray(data[model_1]['CKA_cln'][model_2])
        if len(mat) > 16:
            mat = mat[3::3, 3::3]
        axes[id].imshow(mat)
        axes[id].set_title('CKA clean data ' + beautiful_model_name(model_1) + '\nand ' + beautiful_model_name(model_2))
        axes[id].set_xticks(range(0, len(mat[0]), 2))
        axes[id].set_yticks(range(0, len(mat), 2))
        axes[id].set_xlabel('Blocks ID of ' + beautiful_model_name(model_1))
        axes[id].set_ylabel('Blocks ID of ' + beautiful_model_name(model_2))
    plt.tight_layout()
    plt.savefig('Plots/CKA_t2ts_and_vits_cln.png')
    plt.close()


def plot_CKA_adv(data, model, attacks):
    fig, axes = plt.subplots(nrows=1, ncols=len(attacks), figsize=(12, 5))
    for id, attack in enumerate(attacks):
        mat = np.asarray(data[model]['CKA_adv' + attack])
        if len(mat) > 16:
            mat = mat[3::3, 3::3]
        axes[id].imshow(mat)
        axes[id].set_title('CKA of ' + beautiful_model_name(model) + '\nunder '+ beautiful_attack_name(attack, cln_is_no=True) + ' attack')
        axes[id].set_xticks(range(0, len(mat), 2))
        axes[id].set_yticks(range(0, len(mat), 2))
        axes[id].set_xlabel('Blocks ID')
        axes[id].set_ylabel('Blocks ID')
    plt.tight_layout()
    plt.savefig('Plots/CKA_adversarial_' + model + '.png')
    plt.close()


def table_top1_val(data, attacks, list_models):
    table = []
    headers = ['Model']
    for atk in attacks:
        headers.append(beautiful_attack_name(atk))
    for model in list_models:
        row = [beautiful_model_name(model)]
        for attack in attacks:
            if attack == '_cln':
                row.append(data[model]['Metrics' + attack]['top1'])
            else:
                row.append(data[model]['Metrics_adv' + attack]['top1'])
        table.append(row)
    print(tabulate(table, headers=headers, tablefmt='latex'))


def main():
    data = json.load(open('saves/all_summaries_03-02_14:30.json', 'r'))
    attacks = ['_steps:40_eps:0.001', '_steps:40_eps:0.003', '_steps:40_eps:0.005', '_steps:40_eps:0.01',
               '_steps:1_eps:0.031', '_steps:1_eps:0.062']
    list_models = ['t2t_vit_14_p', 't2t_vit_14_t', 't2t_vit_14_t_doexp05l', 't2t_vit_14_t_donegexp025l', 't2t_vit_14_t_donegexp05l',
                   't2t_vit_14_t_donegexp075l',
                   'vit_base_patch16_224_pretrained', 'vit_base_patch32_224_pretrained', 'vit_base_patch32_224_scratch',
                   'vit_base_patch32_224_donegexp025l', 'vit_base_patch32_224_donegexp025l_finetuned']
    base_models = ['t2t_vit_14_p', 't2t_vit_14_t', 'vit_base_patch32_224_pretrained', 'vit_base_patch16_224_pretrained']
    vit_models = ['vit_base_patch32_224_pretrained', 'vit_base_patch32_224_scratch', 'vit_base_patch32_224_donegexp025l', 'vit_base_patch32_224_donegexp025l_finetuned']
    t2t_models = ['t2t_vit_14_p', 't2t_vit_14_t', 't2t_vit_14_t_doexp05l', 't2t_vit_14_t_donegexp025l', 't2t_vit_14_t_donegexp05l', 't2t_vit_14_t_donegexp075l']

    get_top1_val_base(data, attacks, base_models)
    compare_att_distances_attack_avg_baseModels(data, 'AttDist_cln', base_models)
    compare_att_distances_model_avg(data, 'vit_base_patch32_224_pretrained', attacks)
    compare_att_distances_model_avg(data, 't2t_vit_14_t', attacks)
    CKA_and_attDist_plot(data, 't2t_vit_14_t', 'vit_base_patch32_224_scratch', '_cln')
    CKA_and_attDist_plot(data, 't2t_vit_14_t', 't2t_vit_14_p', '_cln')
    CKA_and_attDist_plot(data, 't2t_vit_14_t', 't2t_vit_14_t_doexp05l', '_cln')
    CKA_and_attDist_plot(data, 't2t_vit_14_t', 't2t_vit_14_t_donegexp025l', '_cln')
    CKA_and_attDist_plot(data, 't2t_vit_14_t', 't2t_vit_14_t_donegexp05l', '_cln')
    CKA_and_attDist_plot(data, 'vit_base_patch32_224_scratch', 'vit_base_patch32_224_donegexp025l', '_cln')
    plot_do_fn()
    compare_att_distances_attack_avg_vitModels(data, 'AttDist_cln', vit_models)
    compare_att_distances_attack_avg_t2tModels(data, 'AttDist_cln', t2t_models)
    get_top1_val_all(data, attacks, list_models)
    plot_CKA_clean(data, ['t2t_vit_14_t', 't2t_vit_14_t', 'vit_base_patch32_224_pretrained'], ['t2t_vit_14_p', 'vit_base_patch32_224_pretrained', 'vit_base_patch16_224_pretrained'])
    plot_CKA_adv(data, 't2t_vit_14_t', ['_steps:40_eps:0.001', '_steps:1_eps:0.031', '_steps:1_eps:0.062'])
    table_top1_val(data, ['_cln'] + attacks, list_models)
    

if __name__ == '__main__':
    main()
