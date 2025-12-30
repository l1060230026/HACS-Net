"""
Train DawNet - Domain Adaptation Network with Hybrid Attention and Whitening
Domain adaptive semantic segmentation based on BIM-to-Scan
"""
import argparse
import os
import torch
import torch.nn.functional as F
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import numpy as np
import time
from data_utils import transform as t
from data_utils.data_util import collate_fn
from data_utils.S3DISDataLoader import S3DISDatasetTrans

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


class LabelMappingDataset:
    """Wrapper dataset that applies label mapping and filters excluded classes"""
    def __init__(self, dataset, label_mapping):
        self.dataset = dataset
        self.label_mapping = label_mapping  # Keep as numpy array
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        coord, label = self.dataset[idx]
        
        # Ensure coord and label are numpy arrays
        if isinstance(coord, torch.Tensor):
            coord = coord.numpy()
        if isinstance(label, torch.Tensor):
            label = label.numpy()
        
        # Apply label mapping
        label_mapped = self.label_mapping[label]
        # Create mask: keep points with labels not equal to -1
        mask = label_mapped >= 0
        coord_filtered = coord[mask]
        label_filtered = label_mapped[mask]
        return coord_filtered, label_filtered


class DualDomainDataset:
    """Dual domain dataset: simultaneously loads source domain (BIM) and target domain (real point cloud) data"""
    def __init__(self, source_dataset, target_dataset):
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        # Use smaller dataset length to avoid excessive repeated sampling (consistent with train_att_hierarchical_gan.py)
        self.length = min(len(source_dataset), len(target_dataset))
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Get data from source domain
        source_idx = idx % len(self.source_dataset)
        source_coord, source_label = self.source_dataset[source_idx]
        
        # Get data from target domain (target domain may not have labels)
        target_idx = idx % len(self.target_dataset)
        target_coord, target_label = self.target_dataset[target_idx]
        
        return source_coord, source_label, target_coord, target_label


def create_label_mapping(exclude_classes):
    """
    Create label mapping, mapping excluded classes to -1, and remapping other classes to continuous indices
    """
    if exclude_classes is None or len(exclude_classes) == 0:
        label_mapping = np.arange(len(classes))
        return label_mapping, classes, class2label, seg_label_to_cat
    
    exclude_indices = [class2label[cls] for cls in exclude_classes if cls in class2label]
    new_classes = [cls for cls in classes if cls not in exclude_classes]
    new_class2label = {cls: i for i, cls in enumerate(new_classes)}
    new_seg_label_to_cat = {}
    for i, cat in enumerate(new_classes):
        new_seg_label_to_cat[i] = cat
    
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
        m.inplace = True


def parse_args():
    parser = argparse.ArgumentParser('DawNet Training')
    parser.add_argument('--model', type=str, default='pointnet2_dawnet', help='model name')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training')
    parser.add_argument('--epoch', default=100, type=int, help='Epoch to run')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate')
    parser.add_argument('--gpu', type=str, default='1', help='GPU to use')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD')
    parser.add_argument('--log_dir', type=str, default='dawnet_bim2scan', help='Log path')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--npoint', type=int, default=40000, help='Point Number')
    parser.add_argument('--step_size', type=int, default=10, help='Decay step for lr decay')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='Decay rate for lr decay')
    parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test')
    
    # Data paths
    parser.add_argument('--source_data_root', type=str, default='data/bim_indoor3d', 
                       help='Source domain (BIM) data root')
    parser.add_argument('--target_data_root', type=str, default='data/stanford_indoor3d', 
                       help='Target domain (Real scan) data root')
    
    # Domain adaptation parameters
    parser.add_argument('--lambda_adv', type=float, default=1.0, 
                       help='Weight for adversarial loss')
    parser.add_argument('--use_whitening', action='store_true', default=True,
                       help='Use ZCA whitening')
    parser.add_argument('--use_attention', action='store_true', default=True,
                       help='Use hybrid attention')
    
    # Data caching
    parser.add_argument('--cache_data', action='store_true', default=False, 
                       help='Cache all data in memory')
    parser.add_argument('--exclude_classes', type=str, nargs='+', 
                       default=['board', 'clutter'], 
                       help='Classes to exclude from training')
    
    return parser.parse_args()


def compute_lambda_p(epoch, max_epoch):
    """
    Compute lambda parameter for domain adversarial training (gradually increasing)
    According to the paper, use progressive adjustment strategy
    """
    p = float(epoch) / max_epoch
    lambda_p = 2. / (1. + np.exp(-10 * p)) - 1
    return lambda_p


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
    logger = logging.getLogger("DawNet")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    # Create label mapping
    exclude_classes = args.exclude_classes if args.exclude_classes else []
    label_mapping, new_classes, new_class2label, new_seg_label_to_cat = create_label_mapping(exclude_classes)
    
    # Update global variables
    global seg_label_to_cat
    seg_label_to_cat = new_seg_label_to_cat
    
    NUM_CLASSES = len(new_classes)
    NUM_POINT = args.npoint
    BATCH_SIZE = args.batch_size
    
    if exclude_classes:
        log_string(f"Excluding classes: {exclude_classes}")
        log_string(f"Training with {NUM_CLASSES} classes: {new_classes}")
    else:
        log_string(f"Training with all {NUM_CLASSES} classes")

    log_string("Loading source domain (BIM) data...")
    log_string(f"Data loading mode: {'cache in memory' if args.cache_data else 'load from disk on-the-fly'}")
    
    # Data augmentation
    train_transform = t.Compose([
        t.RandomRotate(),
        t.RandomFlip(p=0.5),
        t.RandomJitter(sigma=0.01, clip=0.05)
    ])
    
    # Load source domain data (BIM)
    SOURCE_TRAIN_RAW = S3DISDatasetTrans(
        split='train', 
        data_root=args.source_data_root, 
        num_point=NUM_POINT, 
        test_area=args.test_area, 
        sample_rate=1.0, 
        shuffle_index=True, 
        transform=train_transform, 
        cache=args.cache_data
    )
    SOURCE_TRAIN = LabelMappingDataset(SOURCE_TRAIN_RAW, label_mapping)
    
    log_string("Loading target domain (Real scan) data...")
    # Load target domain data (real scan)
    TARGET_TRAIN_RAW = S3DISDatasetTrans(
        split='train', 
        data_root=args.target_data_root, 
        num_point=NUM_POINT, 
        test_area=args.test_area, 
        sample_rate=1.0, 
        shuffle_index=True, 
        transform=train_transform, 
        cache=args.cache_data
    )
    TARGET_TRAIN = LabelMappingDataset(TARGET_TRAIN_RAW, label_mapping)
    
    log_string("Loading test data...")
    # Test data (test on target domain)
    TEST_DATASET_RAW = S3DISDatasetTrans(
        split='test', 
        data_root=args.source_data_root, 
        num_point=NUM_POINT, 
        test_area=args.test_area, 
        sample_rate=1.0, 
        shuffle_index=False, 
        transform=None, 
        cache=args.cache_data
    )
    
    TEST_DATASET = LabelMappingDataset(TEST_DATASET_RAW, label_mapping)

    # Create dual domain dataset
    TRAIN_DATASET = DualDomainDataset(SOURCE_TRAIN, TARGET_TRAIN)
    
    # Custom collate function for dual domain data
    def dual_domain_collate_fn(batch):
        """
        Collate function for dual domain data
        Returns: source_coord, source_label, source_offset, target_coord, target_label, target_offset
        """
        source_coords, source_labels, target_coords, target_labels = [], [], [], []
        
        for s_coord, s_label, t_coord, t_label in batch:
            # Check data type and convert to Tensor
            if isinstance(s_coord, np.ndarray):
                s_coord = torch.from_numpy(s_coord).float()
            else:
                s_coord = s_coord.float()
            
            if isinstance(s_label, np.ndarray):
                s_label = torch.from_numpy(s_label).long()
            else:
                s_label = s_label.long()
            
            if isinstance(t_coord, np.ndarray):
                t_coord = torch.from_numpy(t_coord).float()
            else:
                t_coord = t_coord.float()
            
            if isinstance(t_label, np.ndarray):
                t_label = torch.from_numpy(t_label).long()
            else:
                t_label = t_label.long()
            
            source_coords.append(s_coord)
            source_labels.append(s_label)
            target_coords.append(t_coord)
            target_labels.append(t_label)
        
        # Concatenate source domain data
        source_offset = [0]
        for coord in source_coords:
            source_offset.append(source_offset[-1] + coord.shape[0])
        source_offset = torch.IntTensor(source_offset[1:])
        source_coord = torch.cat(source_coords, 0)
        source_label = torch.cat(source_labels, 0)
        
        # Concatenate target domain data
        target_offset = [0]
        for coord in target_coords:
            target_offset.append(target_offset[-1] + coord.shape[0])
        target_offset = torch.IntTensor(target_offset[1:])
        target_coord = torch.cat(target_coords, 0)
        target_label = torch.cat(target_labels, 0)
        
        return source_coord, source_label, source_offset, target_coord, target_label, target_offset
    
    trainDataLoader = torch.utils.data.DataLoader(
        TRAIN_DATASET, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=2,  # Set to 0 to avoid using shared memory, load data in main process
        pin_memory=False, 
        drop_last=True,
        collate_fn=dual_domain_collate_fn
    )
    
    # Create compatible collate function for test data
    def test_collate_fn(batch):
        """
        Collate function for test data, compatible with numpy array and Tensor
        """
        coords, labels = [], []
        
        for coord, label in batch:
            # Check data type and convert to Tensor
            if isinstance(coord, np.ndarray):
                coord = torch.from_numpy(coord).float()
            else:
                coord = coord.float()
            
            if isinstance(label, np.ndarray):
                label = torch.from_numpy(label).long()
            else:
                label = label.long()
            
            coords.append(coord)
            labels.append(label)
        
        # Compute offset
        offset = [0]
        for coord in coords:
            offset.append(offset[-1] + coord.shape[0])
        offset = torch.IntTensor(offset[1:])
        
        # Concatenate data
        coord = torch.cat(coords, 0)
        label = torch.cat(labels, 0)
        
        return coord, label, offset
    
    testDataLoader = torch.utils.data.DataLoader(
        TEST_DATASET, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=2,  # Set to 0 to avoid using shared memory, load data in main process
        pin_memory=False, 
        collate_fn=test_collate_fn
    )
    
    # Compute class weights
    if exclude_classes:
        original_weights = SOURCE_TRAIN_RAW.labelweights
        keep_indices = [class2label[cls] for cls in new_classes]
        weights = torch.Tensor(original_weights[keep_indices]).cuda()
    else:
        weights = torch.Tensor(SOURCE_TRAIN_RAW.labelweights).cuda()

    log_string("The number of source training data is: %d" % len(SOURCE_TRAIN))
    log_string("The number of target training data is: %d" % len(TARGET_TRAIN))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(experiment_dir))

    classifier = MODEL.get_model(
        NUM_CLASSES, 
        use_whitening=args.use_whitening, 
        use_attention=args.use_attention
    ).cuda()
    criterion = MODEL.get_loss(lambda_adv=args.lambda_adv).cuda()
    classifier.apply(inplace_relu)

    # Try to load pretrained model
    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        classifier.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    # Optimizer
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, classifier.parameters()),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(
            classifier.parameters(), 
            lr=args.learning_rate, 
            momentum=0.9, 
            weight_decay=args.decay_rate
        )

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
        '''Train on dual domains'''
        log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))
        
        # Learning rate adjustment
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # BN momentum adjustment
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        
        # Compute lambda parameter for domain adversarial training
        lambda_p = compute_lambda_p(epoch, args.epoch)
        log_string('Domain adversarial lambda: %f' % lambda_p)
        
        # Training
        num_batches = len(trainDataLoader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        seg_loss_sum = 0
        domain_loss_sum = 0
        classifier = classifier.train()

        loop = tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), leave=True, smoothing=0.9)
        loop.set_description('train')

        for i, (s_coord, s_label, s_offset, t_coord, t_label, t_offset) in loop:
            optimizer.zero_grad()

            # Source domain data
            s_coord = s_coord.cuda(non_blocking=True)
            s_label = s_label.cuda(non_blocking=True)
            s_offset = s_offset.cuda(non_blocking=True)
            
            # Target domain data
            t_coord = t_coord.cuda(non_blocking=True)
            t_label = t_label.cuda(non_blocking=True)
            t_offset = t_offset.cuda(non_blocking=True)

            # Source domain forward pass (segmentation only uses labeled samples from source domain)
            s_seg_pred, s_domain_pred, s_feature = classifier([s_coord, s_coord, s_offset], alpha=lambda_p)
            s_seg_pred = s_seg_pred.contiguous().view(-1, NUM_CLASSES)
            s_label_flat = s_label.view(-1, 1)[:, 0]
            
            # Target domain forward pass (only for domain discrimination, not for segmentation supervision)
            _, t_domain_pred, t_feature = classifier([t_coord, t_coord, t_offset], alpha=lambda_p)
            
            # Segmentation loss only uses source domain labels, implementing unsupervised domain adaptation (UDA)
            seg_pred = s_seg_pred
            seg_target = s_label_flat
            
            # Domain labels: 0 for source domain, 1 for target domain
            s_domain_label = torch.zeros(s_domain_pred.shape[0], dtype=torch.long).cuda()
            t_domain_label = torch.ones(t_domain_pred.shape[0], dtype=torch.long).cuda()
            domain_pred = torch.cat([s_domain_pred, t_domain_pred], dim=0)
            domain_target = torch.cat([s_domain_label, t_domain_label], dim=0)
            
            # Compute loss
            total_loss, seg_loss, domain_loss = criterion(
                seg_pred, seg_target, 
                domain_pred, domain_target, 
                weight=weights
            )
            
            # Check if loss is nan, skip this batch if so
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"Warning: nan/inf loss detected at batch {i}, skipping...")
                optimizer.zero_grad()
                continue
            
            total_loss.backward()
            
            # Gradient clipping to prevent gradient explosion
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=10.0)
            
            optimizer.step()

            # Statistics (only compute segmentation accuracy on source domain)
            s_pred_choice = s_seg_pred.cpu().data.max(1)[1].numpy()
            s_batch_label = s_label_flat.cpu().data.numpy()
            correct = np.sum(s_pred_choice == s_batch_label)
            total_correct += correct
            
            if NUM_POINT < 0:
                total_seen += len(s_batch_label)
                acc = correct / len(s_batch_label)
            else:
                total_seen += (BATCH_SIZE * NUM_POINT)
                acc = correct / (BATCH_SIZE * NUM_POINT)
            
            loss_sum += total_loss.item()
            seg_loss_sum += seg_loss.item()
            domain_loss_sum += domain_loss.item()
            
            loop.set_postfix(
                total_loss=total_loss.item(),
                seg_loss=seg_loss.item(), 
                domain_loss=domain_loss.item(),
                acc=f'{acc * 100:.2f}%'
            )
        
        log_string('Training mean total loss: %f' % (loss_sum / num_batches))
        log_string('Training mean seg loss: %f' % (seg_loss_sum / num_batches))
        log_string('Training mean domain loss: %f' % (domain_loss_sum / num_batches))
        log_string('Training accuracy: %f' % (total_correct / float(total_seen)))

        # Periodically save model
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
        
        '''Evaluate on target domain'''
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
                coord = coord.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
                offset = offset.cuda(non_blocking=True)

                # Only need segmentation prediction
                seg_pred, _, _ = classifier([coord, coord, offset], alpha=0.0)
                pred_val = seg_pred.contiguous().cpu().data.numpy()
                seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

                batch_label = target.cpu().data.numpy()
                target_flat = target.view(-1, 1)[:, 0]
                
                # Compute segmentation loss
                valid_mask = target_flat >= 0
                if valid_mask.sum() > 0:
                    loss = F.nll_loss(seg_pred[valid_mask], target_flat[valid_mask], weight=weights)
                    loss_sum += loss.item()
                
                pred_val = np.argmax(pred_val, 1)
                correct = np.sum((pred_val == batch_label))
                total_correct += correct
                
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

