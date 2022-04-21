from collections import OrderedDict
from timm.utils import *
import numpy as np
import torch
import time


def torch_weighted_mean(tensor, weights):
    return (tensor * weights).sum((1, 2, 3)) / weights.sum((1, 2, 3))


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
