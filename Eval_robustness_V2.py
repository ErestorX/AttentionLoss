from torch.nn.parallel import DistributedDataParallel as NativeDDP
from torch.nn import CrossEntropyLoss
from utils_eval import *
from utils import *
import argparse
import warnings
import torch
import json
import os


parser = argparse.ArgumentParser()
parser.add_argument('-data', default='/data/hugo/ImageNet', type=str)
parser.add_argument('-model', default='vit_small_patch16_224', type=str)
parser.add_argument('-ckpt_file', default=None, type=str)
parser.add_argument('-model_param', default=None, type=str)
parser.add_argument('-b', default=64, type=int)
parser.add_argument('-gpu', default=0, type=int)  # 0 for the RTX A5000 on workstation


def main(args):
    if os.path.exists('output/val/attention_summary.json'):
        with open('output/val/attention_summary.json', 'r') as f:
            all_summaries = json.load(f)
    else:
        all_summaries = {}
        with open('output/val/attention_summary.json', 'w+') as f:
            json.dump(all_summaries, f)
    json_file = 'output/val/attention_summary.json'

    model, experiment_name = load_model_and_make_name(args)
    model = model.eval().cuda()
    if args.distributed:
        model = NativeDDP(model, device_ids=[args.local_rank])
    if experiment_name not in all_summaries.keys():
        all_summaries[experiment_name] = {}

    loader = get_val_loader(args.data, batch_size=args.b)
    loss_fn = CrossEntropyLoss().cuda()

    get_accuracy_and_attention(model, experiment_name, loader, loss_fn, all_summaries[experiment_name], args)
    save_experiment_results(json_file, all_summaries, args.local_rank)

    # attacks = [[1, 0.031], [1, 0.062], [40, 0.001], [40, 0.003], [40, 0.005], [40, 0.01]]
    attacks = [[1, 0.031], [40, 0.003], [40, 0.01]]
    for steps, epsilon in attacks:
        step_size = 0.025 if steps != 1 else 1
        get_attack_accuracy_and_attention(model, experiment_name, loader, loss_fn, all_summaries[experiment_name], args, epsilonMax=epsilon, pgd_steps=steps, step_size=step_size)
        save_experiment_results(json_file, all_summaries, args.local_rank)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    args = parser.parse_args()
    args.local_rank = int(os.environ['LOCAL_RANK'])
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
    else:
        torch.cuda.set_device(0)
    assert args.rank >= 0
    model_ckpt_args_triplets = [
                        ('vit_tiny_patch16_224', None, 'pretrained'),
                        ('vit_small_patch16_224', None, 'pretrained'),
                        ('vit_small_patch32_224', None, 'pretrained'),
                        # ('vit_large_patch16_224', None, 'pretrained'),
                        # ('vit_large_patch32_224', None, 'pretrained'),
                        # ('t2t_vit_14_t', 't2t_vit_14_t', None),
                        # ('t2t_vit_14_p', 't2t_vit_14_p', None),
                        # ('att_t2t_vit_14_t', 'att0.1_t2t_vit_14_t', None),
                        ]
    for (model, ckpt, params) in model_ckpt_args_triplets:
        args.model = model
        args.ckpt_file = ckpt
        args.model_param = params
        try:
            main(args)
        except Exception as e:
            if args.local_rank == 0:
                print(e)
