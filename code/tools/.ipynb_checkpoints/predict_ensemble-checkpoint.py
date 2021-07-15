import argparse
import os
import os.path as osp
import shutil
import tempfile

import mmcv
from mmcv import Config, DictAction
from mmcv.image import tensor2imgs
import torch
import torch.distributed as dist
from mmcv.runner import load_checkpoint, get_dist_info, wrap_fp16_model, init_dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmdet.models.detectors.ensemble_model import EnsembleModel
from mmdet.core import encode_mask_results

def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        # print(data['img_metas'][0].data[0][0]['ori_filename'])
        with torch.no_grad():
            # if i % 100 == 0: print(' ', i)
            result = model(return_loss=False, rescale=not show, **data)
            # print(len(result[0][0]))
            # print(len(result[0][1]))
            
        batch_size = data['img'][0].size(0)

        if show or out_dir:
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None
                model.module.show_result(
                    img_show,
                    result[i],
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)

        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test(model, data_loader, tmpdir=None):
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.append(result)

        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    results = collect_results(results, len(dataset), tmpdir)

    return results


def collect_results(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN,),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    # parser.add_argument('config', help='test config file path')
    parser.add_argument('--cfg_list', type=str, nargs='+')
    parser.add_argument('--checkpoint', type=str, nargs='+')
    parser.add_argument('--out', help='output result file')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show_dir', type=str, help='directory where painted images will be saved')
    parser.add_argument(
        '--show_score_thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
             'useful when you want to format the result to a specific format and '
             'submit it to the test server')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()
    args.cfg_list = ['ensemble_configs/cascade_rcnn_r50_rfp_carafe_sac.py',  # detecotrs_r50
                     'ensemble_configs/cascade_rcnn_s101_dcn_fpn.py',        # s101
                     'ensemble_configs/cascade_rcnn_r2_101_dcn_fpn.py',      # r2_101
                     'ensemble_configs/cascade_rcnn_r101_dcn_fpn.py',        # r101
                     'ensemble_configs/cascade_rcnn_x101_32x4d_dcn_fpn.py',  # x101_32x4d
                     'ensemble_configs/cascade_rcnn_swin_small_fpn.py',      # swin_small
                     'ensemble_configs/cascade_rcnn_swin_base_fpn.py']       # swin_base
    args.checkpoint = ['ensemble_configs/detectors_r50_e20-0bd921e5.pth',
                       'ensemble_configs/s101_20-bd7b757b.pth',
                       'ensemble_configs/r2_101_20-487bd3ea.pth',
                       'ensemble_configs/r101_20-db83ab64.pth',
                       'ensemble_configs/x101_32x4d_20-f11fb360.pth',
                       'ensemble_configs/swin_small_e20-0df8a664.pth',
                       'ensemble_configs/swin_base_e20-b06c4eb6.pth']
    cfg = mmcv.Config.fromfile(args.cfg_list[0])
    
    # 修改nms配置
    cfg.model.test_cfg.rcnn = dict(
                                score_thr=0.001,
                                nms=dict(type='soft_nms', iou_threshold=0.5),
                                max_per_img=100)
    # 修改测试路劲
    cfg.data.test.ann_file = '../data/test_B/testB.json'
    cfg.data.test.img_prefix = '../data/test_B/images'
    
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)  # aug_test不支持batch_size>1
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=4,
        dist=False,
        shuffle=False)

    # build the model and load checkpoint
    models = []
    for config_path, checkpoint_path in zip(args.cfg_list, args.checkpoint):
        print(f"config: {config_path}\\n checkpoint: {checkpoint_path}")
        tmp_cfg = mmcv.Config.fromfile(config_path)
        tmp_cfg.model.pretrained = None
        tmp_cfg.data.test.test_mode = True
        model = build_detector(tmp_cfg.model, train_cfg=None, test_cfg=tmp_cfg.get('test_cfg'))
        checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
        # old versions did not save class info in checkpoints, this walkaround is
        # for backward compatibility
        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = dataset.CLASSES
        models.append(model)
    model = EnsembleModel(models, cfg)
    wrap_fp16_model(model)  # 采用fp16加速

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir, args.show_score_thr)
    else:
        model = MMDistributedDataParallel(model.cuda())
        outputs = multi_gpu_test(model, data_loader, args.tmpdir)

    # if args.format_only:
    dataset.format_results(outputs, jsonfile_prefix='./e-box')

if __name__ == '__main__':
    main()