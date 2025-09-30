# try:
#     from vis import save_occ
# except:
#     print('Load Occupancy Visualization Tools Failed.')
import time, argparse, os.path as osp, os
import torch, numpy as np
import torch.distributed as dist

from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.logging import MMLogger
from mmseg.models import build_segmentor
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")


def pass_print(*args, **kwargs):
    pass


def main(local_rank, args):
    export_npz = bool(args.occ_out_dir)
    if export_npz and (local_rank == 0):
        os.makedirs(args.occ_out_dir, exist_ok=True)

    # global settings
    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # load config
    cfg = Config.fromfile(args.py_config)
    cfg.work_dir = args.work_dir

    # init DDP
    if args.gpus > 1:
        distributed = True
        ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = os.environ.get("MASTER_PORT", "20507")
        hosts = int(os.environ.get("WORLD_SIZE", 1))  # number of nodes
        rank = int(os.environ.get("RANK", 0))  # node id
        gpus = torch.cuda.device_count()  # gpus per node
        print(f"tcp://{ip}:{port}")
        dist.init_process_group(
            backend="nccl", init_method=f"tcp://{ip}:{port}",
            world_size=hosts * gpus, rank=rank * gpus + local_rank)
        world_size = dist.get_world_size()
        cfg.gpu_ids = range(world_size)
        torch.cuda.set_device(local_rank)

        if local_rank != 0:
            import builtins
            builtins.print = pass_print
    else:
        distributed = False
        world_size = 1

    writer = None
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.work_dir, f'{timestamp}.log')
    logger = MMLogger('selfocc', log_file=log_file)
    MMLogger._instance_dict['selfocc'] = logger
    logger.info(f'Config:\n{cfg.pretty_text}')

    # build model
    import model
    from dataset import get_dataloader

    my_model = build_segmentor(cfg.model)
    my_model.init_weights()
    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    logger.info(f'Number of params: {n_parameters}')
    if distributed:
        if cfg.get('syncBN', True):
            my_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(my_model)
            logger.info('converted sync bn.')

        find_unused_parameters = cfg.get('find_unused_parameters', False)
        ddp_model_module = torch.nn.parallel.DistributedDataParallel
        my_model = ddp_model_module(
            my_model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
        raw_model = my_model.module
    else:
        my_model = my_model.cuda()
        raw_model = my_model
    logger.info('done ddp model')

    train_dataset_loader, val_dataset_loader = get_dataloader(
        cfg.train_dataset_config,
        cfg.val_dataset_config,
        cfg.train_loader,
        cfg.val_loader,
        dist=distributed,
        val_only=True)

    # resume and load
    cfg.resume_from = ''

    if osp.exists(osp.join(args.work_dir, 'latest.pth')):
        cfg.resume_from = osp.join(args.work_dir, 'latest.pth')
    if args.resume_from:
        cfg.resume_from = args.resume_from

    logger.info('resume from: ' + cfg.resume_from)
    logger.info('work dir: ' + args.work_dir)

    if cfg.resume_from and osp.exists(cfg.resume_from):
        map_location = 'cpu'
        ckpt = torch.load(cfg.resume_from, map_location=map_location)

        state_dict = ckpt.get("state_dict", ckpt)  # New

        # Load with strict=True. This will raise an error if keys don't match.
        missing_keys, unexpected_keys = raw_model.load_state_dict(state_dict, strict=False)  # New
        logger.info(f'missing keys: {missing_keys}')
        print(f"Missing keys: {missing_keys}")  # New
        logger.info(f"Unexpected keys: {unexpected_keys}")  # New
        print(f"Unexpected keys: {unexpected_keys}")  # New

        print(f'Successfully loaded checkpoint from {cfg.resume_from}')  # New
        logger.info(f'Successfully loaded checkpoint from {cfg.resume_from}')

        # raw_model.load_state_dict(ckpt.get("state_dict", ckpt), strict=True)
        # print(f'successfully resumed.')
    else:
        # If no checkpoint is found, it's better to stop than to evaluate a random model.
        logger.info(f'No checkpoint found at {cfg.resume_from}')
        raise FileNotFoundError(f"No checkpoint found at {cfg.resume_from} or in work_dir.")
    """elif cfg.load_from:
        ckpt = torch.load(cfg.load_from, map_location='cpu')
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        try:
            print(raw_model.load_state_dict(state_dict, strict=False))
        except:
            from misc.checkpoint_util import refine_load_from_sd
            print(raw_model.load_state_dict(
                refine_load_from_sd(state_dict), strict=False))"""

    print_freq = cfg.print_freq
    from misc.metric_util import MeanIoU
    # miou_metric = MeanIoU(
    #   list(range(1, 17)),
    #  17, #17,
    # ['barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    # 'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    # 'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    # 'vegetation'],
    # True, 17, filter_minmax=False)
    miou_metric = MeanIoU(
        list(range(1, 17)),
        17,  # 17,
        ['barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
         'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
         'animal', 'traffic_sign', 'other_vehicle', 'train', 'background',
         'free'],  # different to nuscenes
        True, 17, filter_minmax=False)

    miou_metric.reset()

    my_model.eval()
    os.environ['eval'] = 'true'

    amp = cfg.get('amp', False)
    x_grid, y_grid, z_grid = args.grid_size_occ
    expected_shape = x_grid * y_grid * z_grid

    with torch.no_grad():
        for i_iter_val, data in enumerate(tqdm(val_dataset_loader, desc="[EVAL]")):

            for k in list(data.keys()):
                if isinstance(data[k], torch.Tensor):
                    data[k] = data[k].cuda()
            input_imgs = data.pop('img')

            # result_dict = my_model(imgs=input_imgs, metas=data)

            with torch.cuda.amp.autocast(enabled=amp):
                result_dict = my_model(imgs=input_imgs, metas=data)

            if 'final_occ' in result_dict:

                sample_ids = None
                for key in ('sample_token', 'frame_id', 'lidar_token', 'scene_token'):
                    if key in result_dict and isinstance(result_dict[key], (list, tuple)):
                        sample_ids = result_dict[key]
                        break
                    if key in data and isinstance(data[key], (list, tuple)):
                        sample_ids = data[key]
                        break


                for idx, pred in enumerate(result_dict['final_occ']):
                    pred_occ = pred
                    gt_occ = result_dict['sampled_label'][idx]
                    occ_mask = result_dict['occ_mask'][idx].flatten()

                    miou_metric._after_step(pred_occ, gt_occ, occ_mask)

                    if (i_iter_val % 100) != 0:
                        continue

                    # ---- NPZ export (rank 0 only) ----
                    if export_npz and (local_rank == 0):
                        pred_np = pred_occ.detach().cpu().numpy()
                        gt_np = gt_occ.detach().cpu().numpy() if gt_occ is not None else None
                        m_np = occ_mask.detach().cpu().numpy() if occ_mask is not None else None

                        if pred_np.size != expected_shape:
                            logger.warning(
                                f"[NPZ] size mismatch: pred={pred_np.size}, expected={expected_shape}; saving flat.")
                        else:
                            pred_np = pred_np.reshape(x_grid, y_grid, z_grid, order='C')
                            if gt_np is not None and gt_np.size == expected_shape:
                                gt_np = gt_np.reshape(x_grid, y_grid, z_grid, order='C')
                            if m_np is not None and m_np.size == expected_shape:
                                m_np = m_np.reshape(x_grid, y_grid, z_grid, order='C')

                        # compact dtypes (16 classes fits int16)
                        if pred_np.dtype.kind != 'i':
                            pred_np = pred_np.astype(np.int16)
                        if gt_np is not None and gt_np.dtype.kind != 'i':
                            gt_np = gt_np.astype(np.int16)
                        if m_np is not None:
                            m_np = m_np.astype(np.bool_)

                        stem = f"val_{i_iter_val:06d}_{idx}"
                        if sample_ids is not None:
                            try:
                                sid = sample_ids[idx]
                                # handle tensors/bytes/arrays
                                if torch.is_tensor(sid):
                                    sid = sid.item() if sid.ndim == 0 else sid.cpu().numpy().tolist()
                                if isinstance(sid, (list, tuple)):
                                    sid = "_".join(map(str, sid))
                                if isinstance(sid, bytes):
                                    sid = sid.decode("utf-8", "ignore")
                                stem = str(sid)
                            except Exception:
                                pass
                        out_path = os.path.join(args.occ_out_dir, f"{stem}.npz")

                        payload = {'prediction': pred_np}
                        if gt_np is not None:
                            payload['ground_truth'] = gt_np
                        if m_np is not None:
                            payload['mask'] = m_np

                        np.savez_compressed(out_path, **payload)

                    # if args.vis_occ:
                    #     os.makedirs(os.path.join(args.work_dir, 'vis'), exist_ok=True)
                    #     save_occ(
                    #         os.path.join(args.work_dir, 'vis'),
                    #         pred_occ.reshape(1, 200, 200, 16),
                    #         f'val_{i_iter_val}_pred',
                    #         True, 0)
                    #     save_occ(
                    #         os.path.join(args.work_dir, 'vis'),
                    #         gt_occ.reshape(1, 200, 200, 16),
                    #         f'val_{i_iter_val}_gt',
                    #         True, 0)
                    # breakpoint()

            if i_iter_val % print_freq == 0 and local_rank == 0:
                logger.info('[EVAL] Iter %5d' % (i_iter_val))

    miou, iou2 = miou_metric._after_epoch()
    logger.info(f'mIoU: {miou}, iou2: {iou2}')
    miou_metric.reset()

    if writer is not None:
        writer.close()


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/tpv_lidarseg.py')
    parser.add_argument('--work-dir', type=str, default='./out/tpv_lidarseg')
    parser.add_argument('--resume-from', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--vis-occ', action='store_true', default=False)
    parser.add_argument('--occ_out_dir', type=str, default='')
    parser.add_argument('--occ-shape', type=str, default='750,750,64',
                        help='Occupancy grid shape X,Y,Z (e.g. 750,750,64)')
    args = parser.parse_args()

    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    args.grid_size_occ = tuple(int(x) for x in args.occ_shape.split(','))
    print(args)

    if ngpus > 1:
        torch.multiprocessing.spawn(main, args=(args,), nprocs=args.gpus)
    else:
        main(0, args)
