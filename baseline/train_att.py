"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
from data_utils.S3DISDataLoader import S3DISDatasetTrans
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import time
from data_utils import transform as t
from data_utils.data_util import collate_fn

import gc


class LabelMappingDataset:
    """包装数据集，应用标签映射并过滤被排除的类别"""
    def __init__(self, dataset, label_mapping):
        self.dataset = dataset
        self.label_mapping = torch.from_numpy(label_mapping).long()
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        coord, label = self.dataset[idx]
        # 应用标签映射
        label_mapped = self.label_mapping[label]
        # 创建掩码：保留标签不为-1的点
        mask = label_mapped >= 0
        coord_filtered = coord[mask]
        label_filtered = label_mapped[mask]
        return coord_filtered, label_filtered


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase',
           'board', 'clutter']

class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

def create_label_mapping(exclude_classes):
    """
    创建标签映射，将被排除的类别映射为-1，其他类别重新映射为连续索引
    Args:
        exclude_classes: 要排除的类别名称列表，例如 ['board', 'clutter']
    Returns:
        label_mapping: 原始标签到新标签的映射数组
        new_classes: 保留的类别列表
        new_class2label: 新类别到新标签的映射
        new_seg_label_to_cat: 新标签到类别的映射
    """
    if exclude_classes is None or len(exclude_classes) == 0:
        # 不排除任何类别，返回原始映射
        label_mapping = np.arange(len(classes))
        return label_mapping, classes, class2label, seg_label_to_cat
    
    # 获取要排除的类别索引
    exclude_indices = [class2label[cls] for cls in exclude_classes if cls in class2label]
    
    # 创建保留的类别列表
    new_classes = [cls for cls in classes if cls not in exclude_classes]
    
    # 创建新的类别到标签映射
    new_class2label = {cls: i for i, cls in enumerate(new_classes)}
    
    # 创建新标签到类别的映射
    new_seg_label_to_cat = {}
    for i, cat in enumerate(new_classes):
        new_seg_label_to_cat[i] = cat
    
    # 创建标签映射数组：原始标签 -> 新标签（-1表示被排除）
    label_mapping = np.full(len(classes), -1, dtype=np.int64)
    new_idx = 0
    for orig_idx, cls in enumerate(classes):
        if cls not in exclude_classes:
            label_mapping[orig_idx] = new_idx
            new_idx += 1
    
    return label_mapping, new_classes, new_class2label, new_seg_label_to_cat

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet2_sem_seg_att', help='model name [default: pointnet_sem_seg]')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch', default=50, type=int, help='Epoch to run [default: 32]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default='bim_scan_random_multi', help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int, default=40000, help='Point Number [default: 4096]')
    parser.add_argument('--step_size', type=int, default=10, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 5]')
    parser.add_argument('--data_root', type=str, default='data/bim_scan_random_multi', help='Data root [default: data/bim_indoor3d]')
    parser.add_argument('--cache_data', action='store_true', default=False, help='Cache all data in memory (faster but uses more RAM). If not set, data will be loaded from disk on-the-fly.')
    parser.add_argument('--exclude_classes', type=str, nargs='+', default=['board', 'clutter'], help='Classes to exclude from training, e.g., --exclude_classes board clutter [default: None]')

    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('sem_seg')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    root = args.data_root
    
    # 创建标签映射
    exclude_classes = args.exclude_classes if args.exclude_classes else []
    label_mapping, new_classes, new_class2label, new_seg_label_to_cat = create_label_mapping(exclude_classes)
    
    # 更新全局变量以使用新的类别映射
    global seg_label_to_cat
    seg_label_to_cat = new_seg_label_to_cat
    
    NUM_CLASSES = len(new_classes)
    NUM_POINT = args.npoint # None means unlimited points
    BATCH_SIZE = args.batch_size
    
    if exclude_classes:
        log_string(f"Excluding classes: {exclude_classes}")
        log_string(f"Training with {NUM_CLASSES} classes: {new_classes}")
    else:
        log_string(f"Training with all {NUM_CLASSES} classes")

    print("start loading training data ...")
    log_string(f"Data loading mode: {'cache in memory' if args.cache_data else 'load from disk on-the-fly'}")
    train_transform = t.Compose([
        # t.RandomScale([0.9, 1.1], anisotropic=True),
        t.RandomRotate(),
        # t.RandomShift([0.2, 0.2, 0.0]),
        t.RandomFlip(p=0.5),
        t.RandomJitter(sigma=0.01, clip=0.05)
    ])
    TRAIN_DATASET_RAW = S3DISDatasetTrans(split='train', data_root=root, num_point=NUM_POINT, test_area=args.test_area, sample_rate=1.0, shuffle_index=True, transform=train_transform, cache=args.cache_data)
    TRAIN_DATASET = LabelMappingDataset(TRAIN_DATASET_RAW, label_mapping)
    print("start loading test data ...")
    TEST_DATASET_RAW = S3DISDatasetTrans(split='test', data_root=root, num_point=NUM_POINT, test_area=args.test_area, sample_rate=1.0, shuffle_index=False, transform=None, cache=args.cache_data)
    TEST_DATASET = LabelMappingDataset(TEST_DATASET_RAW, label_mapping)

    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=5,
                                                  pin_memory=False, drop_last=True,
                                                  worker_init_fn=lambda x: np.random.seed(x + int(time.time())), collate_fn=collate_fn)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=5,
                                                 pin_memory=False, collate_fn=collate_fn)
    
    # 重新计算权重：只保留保留类别的权重
    if exclude_classes:
        original_weights = TRAIN_DATASET_RAW.labelweights
        # 只保留保留类别的权重
        keep_indices = [class2label[cls] for cls in new_classes]
        weights = torch.Tensor(original_weights[keep_indices]).cuda()
    else:
        weights = torch.Tensor(TRAIN_DATASET_RAW.labelweights).cuda()

    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('models/pointnet2_utils.py', str(experiment_dir))

    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    criterion = MODEL.get_loss().cuda()
    classifier.apply(inplace_relu)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')

        classifier.load_state_dict(checkpoint['model_state_dict'])

        start_epoch = checkpoint['epoch']

        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        # classifier = classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, classifier.parameters()),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.decay_rate)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    global_epoch = 0
    best_iou = 0

    for epoch in range(start_epoch, args.epoch):
        '''Train on chopped scenes'''
        log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        num_batches = len(trainDataLoader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        classifier = classifier.train()

        loop = tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), leave=True, smoothing=0.9)
        loop.set_description('train')

        for i, (coord, target, offset) in loop:
            optimizer.zero_grad()

            coord, target, offset = coord.cuda(non_blocking=True), target.cuda(non_blocking=True), offset.cuda(non_blocking=True)

            seg_pred, trans_feat = classifier([coord, coord, offset])
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

            batch_label = target.view(-1, 1)[:,  0].cpu().data.numpy()
            target = target.view(-1, 1)[:, 0]
            loss = criterion(seg_pred, target, trans_feat, weights)
            loss.backward()
            optimizer.step()

            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
            correct = np.sum(pred_choice == batch_label)
            total_correct += correct
            # Handle variable point numbers when NUM_POINT is None
            if NUM_POINT < 0:
                total_seen += len(batch_label)
                acc = correct / len(batch_label)
            else:
                total_seen += (BATCH_SIZE * NUM_POINT)
                acc = correct / (BATCH_SIZE * NUM_POINT)
            loss_sum += loss.item()
            loop.set_postfix(train_loss=loss.item(), acc=f'{acc * 100:.2f}%')
        log_string('Training mean loss: %f' % (loss_sum / num_batches))
        log_string('Training accuracy: %f' % (total_correct / float(total_seen)))

        if epoch % 5 == 0:
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')
        
        torch.cuda.empty_cache()
        '''Evaluate on chopped scenes'''
        with torch.no_grad():
            num_batches = len(testDataLoader)
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            labelweights = np.zeros(NUM_CLASSES)
            total_seen_class = [0 for _ in range(NUM_CLASSES)]
            total_correct_class = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
            classifier = classifier.eval()

            log_string('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
            for i, (coord, target, offset) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                coord, target, offset = coord.cuda(non_blocking=True), target.cuda(non_blocking=True), offset.cuda(non_blocking=True)

                seg_pred, trans_feat = classifier([coord, coord, offset])
                pred_val = seg_pred.contiguous().cpu().data.numpy()
                seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

                batch_label = target.cpu().data.numpy()
                target = target.view(-1, 1)[:, 0]
                loss = criterion(seg_pred, target, trans_feat, weights)
                loss_sum += loss.item()
                pred_val = np.argmax(pred_val, 1)
                correct = np.sum((pred_val == batch_label))
                total_correct += correct
                # Handle variable point numbers when NUM_POINT is None
                if NUM_POINT < 0:
                    total_seen += len(batch_label.flatten())
                else:
                    total_seen += (BATCH_SIZE * NUM_POINT)
                tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
                labelweights += tmp

                for l in range(NUM_CLASSES):
                    total_seen_class[l] += np.sum((batch_label == l))
                    total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                    total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))

            labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
            mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float32) + 1e-6))
            log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
            log_string('eval point avg class IoU: %f' % (mIoU))
            log_string('eval point accuracy: %f' % (total_correct / float(total_seen)))
            log_string('eval point avg class acc: %f' % (
                np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float32) + 1e-6))))

            iou_per_class_str = '------- IoU --------\n'
            for l in range(NUM_CLASSES):
                iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f \n' % (
                    seg_label_to_cat[l] + ' ' * (13 + 1 - len(seg_label_to_cat[l])), labelweights[l],
                    total_correct_class[l] / float(total_iou_deno_class[l]))

            log_string(iou_per_class_str)
            log_string('Eval mean loss: %f' % (loss_sum / num_batches))
            log_string('Eval accuracy: %f' % (total_correct / float(total_seen)))

            if mIoU >= best_iou:
                best_iou = mIoU
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'class_avg_iou': mIoU,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                log_string('Saving model....')
            log_string('Best mIoU: %f' % best_iou)
        torch.cuda.empty_cache()

        global_epoch += 1


if __name__ == '__main__':
    args = parse_args()
    main(args)
