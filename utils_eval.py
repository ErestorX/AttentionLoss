from collections import OrderedDict
import torch.nn.functional as F
import torch.nn as nn
from timm.utils import *
import numpy as np
import torch
import time


def torch_weighted_mean(tensor, weights):
    return (tensor * weights).sum((1, 2, 3)) / weights.sum((1, 2, 3))


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def PCGrad(atten_grad, ce_grad, sim, shape):
    pcgrad = atten_grad[sim < 0]
    temp_ce_grad = ce_grad[sim < 0]
    dot_prod = torch.mul(pcgrad, temp_ce_grad).sum(dim=-1)
    dot_prod = dot_prod / torch.norm(temp_ce_grad, dim=-1)
    pcgrad = pcgrad - dot_prod.view(-1, 1) * temp_ce_grad
    atten_grad[sim < 0] = pcgrad
    atten_grad = atten_grad.view(shape)
    return atten_grad


def update_attention_stat(model, qkvs, statistics, batch_idx, patch_size, t2t, performer):
    for block, qkv in qkvs.items():
        if t2t and int(block) < 2:
            num_heads = 1
        else:
            num_heads = model.module.blocks[0].attn.num_heads
        B, N, CCC = qkv.shape
        C = CCC // 3
        if performer and int(block) < 2:
            if int(block) == 0:
                k, q, v = torch.split(qkv, model.module.tokens_to_token.attention1.emb, dim=-1)
                k, q = model.module.tokens_to_token.attention1.prm_exp(
                    k), model.module.tokens_to_token.attention1.prm_exp(q)
            elif int(block) == 1:
                k, q, v = torch.split(qkv, model.module.tokens_to_token.attention2.emb, dim=-1)
                k, q = model.module.tokens_to_token.attention2.prm_exp(
                    k), model.module.tokens_to_token.attention2.prm_exp(q)
            shape_k, shape_q = k.shape, q.shape
            k, q = k.reshape(shape_k[0], 1, shape_k[1], shape_k[2]), q.reshape(shape_q[0], 1, shape_q[1],
                                                                               shape_q[2])
        else:
            qkv = qkv.reshape(B, N, 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
            q, k, _ = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * (num_heads ** -0.5)
        attn = attn.softmax(dim=-1).permute(1, 0, 2, 3)
        N = N - 1
        attn = attn[:, :, 1:, 1:]
        vect = torch.arange(N).reshape((1, N))
        dist_map = torch.sqrt((torch.abs((vect - torch.transpose(vect, 0, 1))) % N ** 0.5) ** 2 + (
                (vect - torch.transpose(vect, 0, 1)) // N ** 0.5) ** 2)
        if t2t and int(block) == 0:
            patch_size = patch_size / 4
        elif t2t and int(block) == 1:
            patch_size = patch_size / 2
        val = torch.as_tensor(dist_map).cuda().unsqueeze(0).repeat(num_heads, B, 1, 1) * patch_size
        weights = attn
        avg = torch_weighted_mean(val, weights)
        std = torch.sqrt(torch_weighted_mean((val - avg.reshape(num_heads, 1, 1, 1)) ** 2, weights))
        if block not in statistics.keys():
            statistics[block] = {'avgHead': avg.cpu().numpy().tolist(),
                                 'stdHead': std.cpu().numpy().tolist()}
        else:
            statistics[block]['avgHead'] = (np.add(np.asarray(statistics[block]['avgHead']) * batch_idx,
                                                   avg.cpu().numpy()) / (batch_idx + 1)).tolist()
            statistics[block]['stdHead'] = (np.add(np.asarray(statistics[block]['stdHead']) * batch_idx,
                                                   std.cpu().numpy()) / (batch_idx + 1)).tolist()
    return statistics


def get_accuracy_and_attention(model, name_model, loader, loss_fn, summary, args):
    if 'Metrics_cln' not in summary.keys() or 'AttDist_cln' not in summary.keys():
        if args.local_rank == 0:
            print('\t---Starting validation on clean DS---')
        batch_time_m = AverageMeter()
        losses_m = AverageMeter()
        top1_m = AverageMeter()
        top5_m = AverageMeter()

        def get_features(name):
            def hook(model, input, output):
                qkvs[name] = output.detach()

            return hook

        index = 0
        t2t = 'T2T' in name_model
        performer = t2t and '-p' in name_model
        if t2t:
            index = 2
            if performer:
                model.module.tokens_to_token.attention1.kqv.register_forward_hook(get_features('0'))
                model.module.tokens_to_token.attention2.kqv.register_forward_hook(get_features('1'))
            else:
                model.module.tokens_to_token.attention1.attn.qkv.register_forward_hook(get_features('0'))
                model.module.tokens_to_token.attention2.attn.qkv.register_forward_hook(get_features('1'))

        for block_id, block in enumerate(model.module.blocks):
            block.attn.qkv.register_forward_hook(get_features(str(block_id + index)))

        patch_size = 32 if '32' in name_model else 16
        model.eval()
        qkvs = {}
        statistics = {}
        end = time.time()
        last_idx = len(loader) - 1

        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            output = model(input)
            statistics = update_attention_stat(model, qkvs, statistics, batch_idx, patch_size, t2t, performer)
            if isinstance(output, (tuple, list)):
                output = output[0]

            loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data
            torch.cuda.synchronize()
            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if (last_batch or batch_idx % 100 == 0) and args.local_rank == 0:
                log_name = 'Clean'
                print('{0}: [{1:>4d}/{2}]  Acc@1: {top1.avg:>7.4f}'.format(log_name, batch_idx, last_idx,
                                                                           batch_time=batch_time_m,
                                                                           top1=top1_m))

        summary['Metrics_cln'] = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])
        summary['AttDist_cln'] = list(statistics.values())
        avgHead = []
        stdHead = []
        for block in summary['AttDist_cln']:
            avgHead.append(block['avgHead'])
            stdHead.append(block['stdHead'])
        summary['AttDist_cln'] = {'avgHead': avgHead, 'stdHead': stdHead}


def get_attack_accuracy_and_attention(model, name_model, loader, loss_fn, summary, args, epsilonMax=0.062, pgd_steps=1, step_size=1):
    key_metrics = '_'.join(['Metrics_adv', 'steps:' + str(pgd_steps), 'eps:' + str(epsilonMax)])
    key_attDist = '_'.join(['AttDist_adv', 'steps:' + str(pgd_steps), 'eps:' + str(epsilonMax)])
    if key_metrics not in summary.keys() or key_attDist not in summary.keys():
        if args.local_rank == 0:
            print('\t---Starting validation on DS attacked: steps={} epsilon={}---'.format(pgd_steps, epsilonMax))
        batch_time_m = AverageMeter()
        losses_m = AverageMeter()
        top1_m = AverageMeter()
        top5_m = AverageMeter()

        def get_features(name):
            def hook(model, input, output):
                qkvs[name] = output.detach()

            return hook

        index = 0
        t2t = 'T2T' in name_model
        performer = t2t and '-p' in name_model
        if t2t:
            index = 2
            if performer:
                model.module.tokens_to_token.attention1.kqv.register_forward_hook(get_features('0'))
                model.module.tokens_to_token.attention2.kqv.register_forward_hook(get_features('1'))
            else:
                model.module.tokens_to_token.attention1.attn.qkv.register_forward_hook(get_features('0'))
                model.module.tokens_to_token.attention2.attn.qkv.register_forward_hook(get_features('1'))

        for block_id, block in enumerate(model.module.blocks):
            block.attn.qkv.register_forward_hook(get_features(str(block_id + index)))

        patch_size = 32 if '32' in name_model else 16
        model.eval()
        qkvs = {}
        statistics = {}
        end = time.time()
        last_idx = len(loader) - 1

        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            input_orig = input.clone()
            for _ in range(pgd_steps):
                input.requires_grad = True
                output = model(input)
                model.zero_grad()
                cost = loss_fn(output, target)
                grad = torch.autograd.grad(cost, input, retain_graph=False, create_graph=False)[0]
                input = input + step_size * grad.sign()
                input = input_orig + torch.clamp(input - input_orig, -epsilonMax, epsilonMax)
                input = torch.clamp(input, -1, 1).detach()

            if pgd_steps == 250:
                mu = torch.tensor(args.mu).view(3, 1, 1).cuda()
                std = torch.tensor(args.std).view(3, 1, 1).cuda()
                criterion = nn.CrossEntropyLoss().cuda()

                patch_num_per_line = int(input.size(-1) / patch_size)
                delta = torch.zeros_like(input).cuda()
                delta.requires_grad = True
                out, atten = model(input + delta)
                atten_layer = atten[4].mean(dim=1)
                if 'DeiT' in args.network:
                    atten_layer = atten_layer.mean(dim=-2)[:, 1:]
                else:
                    atten_layer = atten_layer.mean(dim=-2)
                max_patch_index = atten_layer.argsort(descending=True)[:, :args.num_patch]
                mask = torch.zeros([input.size(0), 1, input.size(2), input.size(3)]).cuda()
                for j in range(input.size(0)):
                    index_list = max_patch_index[j]
                    for index in index_list:
                        row = (index // patch_num_per_line) * patch_size
                        column = (index % patch_num_per_line) * patch_size
                        mask[j, :, row:row + patch_size, column:column + patch_size] = 1
                max_patch_index_matrix = max_patch_index[:, 0]
                max_patch_index_matrix = max_patch_index_matrix.repeat(197, 1)
                max_patch_index_matrix = max_patch_index_matrix.permute(1, 0)
                max_patch_index_matrix = max_patch_index_matrix.flatten().long()
                delta = (torch.rand_like(input) - mu) / std
                delta.data = clamp(delta, (0 - mu) / std, (1 - mu) / std)
                original_img = input.clone()
                input = torch.mul(input, 1 - mask)
                input = torch.mul(input, 1 - mask)
                delta = delta.cuda()
                delta.requires_grad = True

                opt = torch.optim.Adam([delta], lr=args.attack_learning_rate)
                scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.95)

                for _ in range(pgd_steps):
                    model.zero_grad()
                    opt.zero_grad()
                    output, atten = model(input + torch.mul(delta, mask))
                    loss = criterion(output, target)
                    grad = torch.autograd.grad(loss, delta, retain_graph=True)[0]
                    ce_loss_grad_temp = grad.view(input.size(0), -1).detach().clone()

                    # Attack the first 6 layers' Attn
                    range_list = range(len(atten) // 2)
                    for atten_num in range_list:
                        if atten_num == 0:
                            continue
                        atten_map = atten[atten_num]
                        atten_map = atten_map.mean(dim=1)
                        atten_map = atten_map.view(-1, atten_map.size(-1))
                        atten_map = -torch.log(atten_map)
                        if 'DeiT' in args.network:
                            atten_loss = F.nll_loss(atten_map, max_patch_index_matrix + 1)
                        else:
                            atten_loss = F.nll_loss(atten_map, max_patch_index_matrix)

                        atten_grad = torch.autograd.grad(atten_loss, delta, retain_graph=True)[0]

                        atten_grad_temp = atten_grad.view(input.size(0), -1)
                        cos_sim = F.cosine_similarity(atten_grad_temp, ce_loss_grad_temp, dim=1)

                        '''PCGrad'''
                        atten_grad = PCGrad(atten_grad_temp, ce_loss_grad_temp, cos_sim, grad.shape)
                        grad += atten_grad * args.atten_loss_weight
                        opt.zero_grad()
                        delta.grad = -grad
                        opt.step()
                        scheduler.step()
                        delta.data = clamp(delta, (0 - mu) / std, (1 - mu) / std)

                input = original_img + torch.mul(delta, mask)


            output = model(input)
            statistics = update_attention_stat(model, qkvs, statistics, batch_idx, patch_size, t2t, performer)
            if isinstance(output, (tuple, list)):
                output = output[0]

            loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data
            torch.cuda.synchronize()
            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if (last_batch or batch_idx % 100 == 0) and args.local_rank == 0:
                log_name = 'Clean'
                print('{0}: [{1:>4d}/{2}]  Acc@1: {top1.avg:>7.4f}'.format(log_name, batch_idx, last_idx,
                                                                           batch_time=batch_time_m,
                                                                           top1=top1_m))

        summary[key_metrics] = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])
        summary[key_attDist] = list(statistics.values())
        avgHead = []
        stdHead = []
        for block in summary[key_attDist]:
            avgHead.append(block['avgHead'])
            stdHead.append(block['stdHead'])
        summary[key_attDist] = {'avgHead': avgHead, 'stdHead': stdHead}
