import time, argparse, os.path as osp, os
import torch, numpy as np
import torch.distributed as dist
from copy import deepcopy

import mmcv
from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.optim import build_optim_wrapper
from mmengine.logging import MMLogger
from mmengine.utils import symlink
from mmseg.models import build_segmentor
from timm.scheduler import CosineLRScheduler, MultiStepLRScheduler

import warnings
warnings.filterwarnings("ignore")


def pass_print(*args, **kwargs):
    pass

def main(local_rank, args):
    # global settings
    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # load config
    cfg = Config.fromfile(args.py_config)
    cfg.work_dir = args.work_dir

    target_idx = 0
    cfg.train_dataset_config.setdefault('vis_indices', [target_idx])
    cfg.train_dataset_config.pop('num_samples', None)

    cfg.val_dataset_config.setdefault('vis_indices', [target_idx])
    cfg.val_dataset_config.pop('num_samples', None)

    # --- make loaders tiny & deterministic ---
    cfg.train_loader.update(dict(batch_size=1, shuffle=False, num_workers=0, persistent_workers=False))
    cfg.val_loader.update(dict(batch_size=1, shuffle=False, num_workers=0, persistent_workers=False))

    cfg.eval_every_epochs = 10 ** 9

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
    
    if local_rank == 0:
        os.makedirs(args.work_dir, exist_ok=True)
        os.makedirs(osp.join(args.work_dir, "predictions"), exist_ok=True)
        cfg.dump(osp.join(args.work_dir, osp.basename(args.py_config)))
        from misc.tb_wrapper import WrappedTBWriter
        writer = WrappedTBWriter('selfocc', log_dir=osp.join(args.work_dir, 'tf'))
        WrappedTBWriter._instance_dict['selfocc'] = writer
    else:
        writer = None
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.work_dir, f'{timestamp}.log')
    logger = MMLogger('selfocc', log_file=log_file)
    MMLogger._instance_dict['selfocc'] = logger

    if local_rank == 0:
        logger.info("--- GPU Detection Check ---")
        logger.info(f"torch.cuda.device_count() sees {args.gpus} GPUs.")
        logger.info(f"DDP world size is {world_size}.")
        logger.info("--- End Check ---")
    
    logger.info(f'Config:\n{cfg.pretty_text}')

    # build model
    import model
    from dataset import get_dataloader
    from loss import OPENOCC_LOSS

    my_model = build_segmentor(cfg.model)
    my_model.init_weights()
    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    logger.info(f'Number of params: {n_parameters}')

    logger.info(f'Params require grad: {[n for n, p in my_model.named_parameters() if p.requires_grad]}')
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

    import torch.nn as nn
    def freeze_bn(m):
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()

    train_dataset_loader, val_dataset_loader = get_dataloader(
        cfg.train_dataset_config,
        cfg.val_dataset_config,
        cfg.train_loader,
        cfg.val_loader,
        dist=distributed,
        iter_resume=args.iter_resume)

    # get optimizer, loss, scheduler
    optimizer = build_optim_wrapper(my_model, cfg.optimizer)
    loss_func = OPENOCC_LOSS.build(cfg.loss).cuda()
    max_num_epochs = cfg.max_epochs
    if cfg.get('multisteplr', False):
        scheduler = MultiStepLRScheduler(
            optimizer,
            **cfg.multisteplr_config
        )
    else:
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=len(train_dataset_loader) * max_num_epochs,
            lr_min=cfg.optimizer["optimizer"]["lr"] * cfg.get("min_lr_ratio", 0.1), #1e-6,
            warmup_t=cfg.get('warmup_iters', 500),
            warmup_lr_init=1e-6,
            t_in_epochs=False)
    amp = cfg.get('amp', False)
    if amp:
        scaler = torch.cuda.amp.GradScaler()
        os.environ['amp'] = 'true'
    else:
        os.environ['amp'] = 'false'
    
    # resume and load
    epoch = 0
    global_iter = 0
    last_iter = 0

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
        print(raw_model.load_state_dict(ckpt['state_dict'], strict=False))
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        epoch = ckpt['epoch']
        global_iter = ckpt['global_iter']
        last_iter = ckpt['last_iter'] if 'last_iter' in ckpt else 0
        if hasattr(train_dataset_loader.sampler, 'set_last_iter'):
            train_dataset_loader.sampler.set_last_iter(last_iter)
        print(f'successfully resumed from epoch {epoch}')
    elif cfg.load_from:
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
                refine_load_from_sd(state_dict), strict=False))
        
    # training
    print_freq = cfg.print_freq
    first_run = True
    grad_accumulation = args.gradient_accumulation
    grad_norm = 0
    from misc.metric_util import MeanIoU
    miou_metric = MeanIoU(
        list(range(1, 17)),
        17, #17,
        ['barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
         'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
         'animal', 'traffic_sign', 'other_vehicle', 'train', 'background',
         'free'], # different to nuscenes
         True, 17, filter_minmax=False)
    miou_metric.reset()

    save_dir = osp.join(args.work_dir, "predictions")

    while epoch < max_num_epochs:
        my_model.train()
        my_model.apply(freeze_bn)
        os.environ['eval'] = 'false'
        if hasattr(train_dataset_loader.sampler, 'set_epoch'):
            train_dataset_loader.sampler.set_epoch(epoch)
        loss_list = []
        time.sleep(1)
        data_time_s = time.time()
        time_s = time.time()
        for i_iter, data in enumerate(train_dataset_loader):
            if first_run:
                i_iter = i_iter + last_iter

            # Create a deepcopy of data for saving logic later, to preserve original tensors
            #data_for_saving = deepcopy(data)

            for k in list(data.keys()):
                if isinstance(data[k], torch.Tensor):
                    data[k] = data[k].cuda()
            input_imgs = data.pop('img')            
            data_time_e = time.time()

            with torch.cuda.amp.autocast(amp):
                # forward + backward + optimize
                result_dict = my_model(imgs=input_imgs, metas=data, global_iter=global_iter)

                ###############################################################
                for key, value in result_dict.items():
                    if isinstance(value, torch.Tensor) and torch.isnan(value).any():
                        logger.error(f"!!! NaN detected in model output: result_dict['{key}'] !!!")
                        # Consider adding exit() here to stop on first detection
                        # exit()
                ################################################################

                loss_input = {
                    'metas': data,
                    'global_iter': global_iter
                }
                for loss_input_key, loss_input_val in cfg.loss_input_convertion.items():
                    loss_input.update({
                        loss_input_key: result_dict[loss_input_val]})
                loss, loss_dict = loss_func(loss_input)
                loss = loss / grad_accumulation

            # -------------> ADD THIS BLOCK <-------------
            # Check the final loss value before backward pass
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"!!! Invalid loss detected (NaN or Inf): {loss.item()}. Halting. !!!")
                # Dumping tensors for debugging
                # torch.save(result_dict, osp.join(args.work_dir, "nan_debug_result_dict.pth"))
                # torch.save(loss_input, osp.join(args.work_dir, "nan_debug_loss_input.pth"))
                exit()
            # ----------------------------------------------

            if not amp:
                loss.backward()
                if (global_iter + 1) % grad_accumulation == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(my_model.parameters(), cfg.grad_max_norm)
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                # -------------> ADD THIS BLOCK <-------------
                # Check the final loss value before backward pass (AMP version)
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.error(f"!!! Invalid loss detected (NaN or Inf) in AMP: {loss.item()}. Halting. !!!")
                    exit()
                # ----------------------------------------------

                scaler.scale(loss).backward()
                if (global_iter + 1) % grad_accumulation == 0:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(my_model.parameters(), cfg.grad_max_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

            loss_list.append(loss.detach().cpu().item())
            scheduler.step_update(global_iter)
            time_e = time.time()

            global_iter += 1
            if i_iter % print_freq == 0 and local_rank == 0:
                lr = max([p['lr'] for p in optimizer.param_groups])
                # lr = optimizer.param_groups[0]['lr']
                logger.info('[TRAIN] Epoch %d Iter %5d/%d: Loss: %.3f (%.3f), grad_norm: %.3f, lr: %.7f, time: %.3f (%.3f)'%(
                    epoch, i_iter, len(train_dataset_loader), 
                    loss.item(), np.mean(loss_list), grad_norm, lr,
                    time_e - time_s, data_time_e - data_time_s))

                if writer is not None: # Added for logging
                    writer.add_scalar('Train/loss_step', loss.item(), global_iter)
                    writer.add_scalar('Train/learning_rate', lr, global_iter)

                detailed_loss = []
                for loss_name, loss_value in loss_dict.items():
                    detailed_loss.append(f'{loss_name}: {loss_value:.5f}')
                detailed_loss = ', '.join(detailed_loss)
                logger.info(detailed_loss)
                loss_list = []
            data_time_s = time.time()
            time_s = time.time()

            if (epoch + 1) % 50 == 0 and local_rank == 0:
                pred = result_dict['final_occ'][0].detach()  # [N]
                gt = result_dict['sampled_label'][0].detach()  # [N]
                mask = result_dict['occ_mask'][0].flatten().detach()  # [N]

                pred = pred.long()
                gt = gt.long()
                mask = mask.bool()

                miou_metric.reset()
                miou_metric._after_step(pred, gt, mask)
                miou, iou2 = miou_metric._after_epoch()
                logger.info(f'[TRAIN quick] Epoch {epoch + 1}: mIoU={miou:.4f}, iou2={iou2:.4f}')
                if writer is not None:
                    writer.add_scalar('TrainQuick/mIoU', miou, epoch + 1)
                    writer.add_scalar('TrainQuick/iou2', iou2, epoch + 1)

            # Save every 200 epochs or on the very last epoch
            if (epoch % 200 == 0) or (epoch == max_num_epochs - 1):
                if local_rank == 0:
                    logger.info(f"--- Saving prediction at epoch {epoch} ---")
                    filename = osp.join(save_dir, f'epoch_{epoch:04d}.npz')

                    occ_mask_3d = result_dict['occ_mask'][0].detach().bool()  # [Z,Y,X]
                    pred_ = result_dict['final_occ'][0].detach().long()  # [N] or flat [Z*Y*X]
                    gt_ = result_dict['sampled_label'][0].detach().long()  # [N] or [Z,Y,X]

                    assert occ_mask_3d.ndim == 3, f"occ_mask must be 3D [Z,Y,X], got {occ_mask_3d.shape}"
                    num_mask = int(occ_mask_3d.sum().item())
                    grid_num = int(occ_mask_3d.numel())

                    EMPTY = 17  # empty/unknown label

                    def to_dense(labels, mask3d):
                        """Return dense [Z,Y,X] labels given possible input forms."""
                        if labels.ndim == 3:
                            # already dense
                            return labels
                        labels = labels.contiguous()
                        n = labels.numel()
                        if n == num_mask:
                            # masked vector -> fill EMPTY and write into mask
                            dense = torch.full(mask3d.shape, EMPTY, dtype=torch.long, device=labels.device)
                            dense[mask3d] = labels
                            return dense
                        if n == grid_num:
                            # dense-flat -> reshape to grid (assumes C-order flatten/view)
                            return labels.view(mask3d.shape)
                        raise ValueError(f"Unexpected labels shape/length: {tuple(labels.shape)} "
                                         f"(mask sum={num_mask}, grid num={grid_num})")

                    pred_dense = to_dense(pred_, occ_mask_3d).contiguous().cpu().numpy().astype(np.uint8)
                    gt_dense = to_dense(gt_, occ_mask_3d).contiguous().cpu().numpy().astype(np.uint8)
                    valid_mask = occ_mask_3d.cpu().numpy().astype(np.bool_)

                    # Pull from cfg if available; otherwise fall back to defaults
                    vx = cfg.get('voxel_size', [0.4, 0.4, 0.4])
                    if isinstance(vx, (int, float)):
                        vx = [float(vx), float(vx), float(vx)]
                    pc_rng = cfg.get('pc_range', [-40, -40, -1, 40, 40, 5.4])

                    np.savez_compressed(
                        filename,
                        prediction=pred_dense,  # [Z,Y,X]
                        ground_truth=gt_dense,  # [Z,Y,X]
                        valid_mask=valid_mask,  # [Z,Y,X]
                        axes_order="XYZ",
                        grid_shape=np.array(pred_dense.shape, np.int32),
                        voxel_size=np.asarray(vx, dtype=np.float32),
                        pc_range=np.asarray(pc_rng, dtype=np.float32),
                        epoch=np.array([epoch], np.int32),
                        empty_label=np.array([EMPTY], np.int32),
                    )

                    # Invariants (ok to comment out later)
                    assert pred_dense.shape == tuple(occ_mask_3d.shape)
                    assert (pred_dense[~valid_mask] == EMPTY).all(), "outside-mask must be EMPTY"

            if args.iter_resume and False:
                if (i_iter + 1) % 50 == 0 and local_rank == 0:
                    dict_to_save = {
                        'state_dict': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch,
                        'global_iter': global_iter,
                        'last_iter': i_iter + 1,
                    }
                    save_file_name = os.path.join(os.path.abspath(args.work_dir), 'iter.pth')
                    torch.save(dict_to_save, save_file_name)
                    dst_file = osp.join(args.work_dir, 'latest.pth')
                    symlink(save_file_name, dst_file)
                    logger.info(f'iter ckpt {i_iter + 1} saved!')
        
        # save checkpoint
        if local_rank == 0 and False:
            dict_to_save = {
                'state_dict': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch + 1,
                'global_iter': global_iter,
            }
            save_file_name = os.path.join(os.path.abspath(args.work_dir), f'epoch_{epoch+1}.pth')
            torch.save(dict_to_save, save_file_name)
            dst_file = osp.join(args.work_dir, 'latest.pth')
            symlink(save_file_name, dst_file)

        epoch += 1
        first_run = False
        
        # eval
        if epoch % cfg.get('eval_every_epochs', 1) != 0:
            continue
        my_model.eval()
        os.environ['eval'] = 'true'
        val_loss_list = []

        with torch.no_grad():
            for i_iter_val, data in enumerate(val_dataset_loader):
                for k in list(data.keys()):
                    if isinstance(data[k], torch.Tensor):
                        data[k] = data[k].cuda()
                input_imgs = data.pop('img')
                
                with torch.cuda.amp.autocast(amp):
                    result_dict = my_model(imgs=input_imgs, metas=data)

                    loss_input = {
                        'metas': data,
                        'global_iter': global_iter
                    }
                    for loss_input_key, loss_input_val in cfg.loss_input_convertion.items():
                        loss_input.update({
                            loss_input_key: result_dict[loss_input_val]})
                    loss, loss_dict = loss_func(loss_input)
                
                if 'final_occ' in result_dict:
                    for idx, pred in enumerate(result_dict['final_occ']):
                        pred_occ = pred
                        gt_occ = result_dict['sampled_label'][idx]
                        occ_mask = result_dict['occ_mask'][idx].flatten()
                        miou_metric._after_step(pred_occ, gt_occ, occ_mask)
                
                val_loss_list.append(loss.detach().cpu().numpy())
                if i_iter_val % print_freq == 0 and local_rank == 0:
                    logger.info('[EVAL] Epoch %d Iter %5d: Loss: %.3f (%.3f)'%(
                        epoch, i_iter_val, loss.item(), np.mean(val_loss_list)))
                    detailed_loss = []
                    for loss_name, loss_value in loss_dict.items():
                        detailed_loss.append(f'{loss_name}: {loss_value:.5f}')
                    detailed_loss = ', '.join(detailed_loss)
                    logger.info(detailed_loss)
                        
        miou, iou2 = miou_metric._after_epoch()
        logger.info(f'mIoU: {miou}, iou2: {iou2}')
        logger.info('Current val loss is %.3f' % (np.mean(val_loss_list)))

        if writer is not None: # Add logging
            writer.add_scalar('Eval/loss_epoch', np.mean(val_loss_list), epoch)
            writer.add_scalar('Eval/mIoU', miou, epoch)
            writer.add_scalar('Eval/iou2', iou2, epoch)

            if hasattr(miou_metric, 'iou') and hasattr(miou_metric, 'label_str'):
                for i, iou_class in enumerate(miou_metric.iou):
                    # Use 'label_str' as defined in the MeanIoU class
                    class_name = miou_metric.label_str[i]
                    writer.add_scalar(f'IoU/{class_name}', iou_class * 100, epoch)

        miou_metric.reset()
    
    if writer is not None:
        writer.close()
        

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/tpv_lidarseg.py')
    parser.add_argument('--work-dir', type=str, default='./out/tpv_lidarseg')
    parser.add_argument('--resume-from', type=str, default='')
    parser.add_argument('--iter-resume', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gradient-accumulation', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='nuscenes')
    args = parser.parse_args()
    
    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    print(args)

    if ngpus > 1:
        torch.multiprocessing.spawn(main, args=(args,), nprocs=args.gpus)
    else:
        main(0, args)
