"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
from data_utils.S3DISDataLoader import ScannetDatasetWholeScene
from data_utils.indoor3d_util import g_label2color
from data_utils.data_util import voxelize
import torch
import logging
from pathlib import Path
import sys
import importlib
from tqdm import tqdm
import provider
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase',
           'board', 'clutter']


g_easy_view_labels = [0, 13]

# classes = ['ceiling', 'floor', 'wall', 'door', 'window', 'table', 'chair', 'cabinet', 'shelf',
#            'bin', 'box', 'board', 'screen', 'clutter']

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
        reverse_mapping: 新标签到原始标签的映射数组（用于测试时将预测映射回原始空间）
        new_classes: 保留的类别列表
        new_class2label: 新类别到新标签的映射
        new_seg_label_to_cat: 新标签到类别的映射
    """
    if exclude_classes is None or len(exclude_classes) == 0:
        # 不排除任何类别，返回原始映射
        label_mapping = np.arange(len(classes))
        reverse_mapping = np.arange(len(classes))
        return label_mapping, reverse_mapping, classes, class2label, seg_label_to_cat
    
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
    
    # 创建反向映射：新标签 -> 原始标签
    reverse_mapping = np.full(len(new_classes), -1, dtype=np.int64)
    for orig_idx, cls in enumerate(classes):
        if cls not in exclude_classes:
            new_idx = label_mapping[orig_idx]
            reverse_mapping[new_idx] = orig_idx
    
    return label_mapping, reverse_mapping, new_classes, new_class2label, new_seg_label_to_cat

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in testing [default: 32]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=40000, help='point number [default: 4096]')
    parser.add_argument('--voxel_size', type=float, default=0.04, help='voxel size')
    parser.add_argument('--log_dir', type=str, default='att_hierarchical_gan', help='experiment root')
    parser.add_argument('--visual', action='store_true', default=True, help='visualize result [default: False]')
    parser.add_argument('--test_area', type=int, default=5, help='area for testing, option: 1-6 [default: 5]')
    parser.add_argument('--num_votes', type=int, default=3, help='aggregate segmentation scores with voting [default: 5]')
    parser.add_argument('--exclude_classes', type=str, nargs='+', default=['board', 'clutter'], help='Classes excluded during training, e.g., --exclude_classes board clutter [default: board clutter]')
    return parser.parse_args()


def add_vote(vote_label_pool, point_idx, pred_label, weight):
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            if weight[b, n] != 0 and not np.isinf(weight[b, n]):
                vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool


def input_normalize(coord):
    coord_min = np.min(coord, 0)
    coord -= coord_min
    return coord


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'log/sem_seg/' + args.log_dir
    visual_dir = experiment_dir + '/visual/'
    visual_dir = Path(visual_dir)
    visual_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    # 创建标签映射（与训练时一致）
    exclude_classes = args.exclude_classes if args.exclude_classes else []
    label_mapping, reverse_mapping, new_classes, new_class2label, new_seg_label_to_cat = create_label_mapping(exclude_classes)
    
    # 更新全局变量以使用新的类别映射（用于显示）
    global seg_label_to_cat
    seg_label_to_cat = new_seg_label_to_cat
    
    # 模型输出的类别数（与训练时一致）
    NUM_CLASSES_MODEL = len(new_classes)
    # 原始类别数（用于ground truth）
    NUM_CLASSES_ORIGINAL = len(classes)
    
    BATCH_SIZE = args.batch_size
    NUM_POINT = args.num_point
    
    if exclude_classes:
        log_string(f"Excluded classes during training: {exclude_classes}")
        log_string(f"Model outputs {NUM_CLASSES_MODEL} classes: {new_classes}")
        log_string(f"Ground truth has {NUM_CLASSES_ORIGINAL} classes: {classes}")
    else:
        log_string(f"Using all {NUM_CLASSES_MODEL} classes")

    root = 'data/bim_scan/'

    TEST_DATASET_WHOLE_SCENE = ScannetDatasetWholeScene(root, split='test', test_area=args.test_area, block_points=NUM_POINT)
    log_string("The number of test data is: %d" % len(TEST_DATASET_WHOLE_SCENE))

    '''MODEL LOADING'''
    # 从实验目录根目录查找模型文件（训练时复制的模型文件）
    # 排除 pointnet2_utils.py，查找其他 .py 文件作为模型文件
    experiment_dir_path = Path(experiment_dir)
    model_files = [f for f in experiment_dir_path.glob('*.py') 
                   if f.name != 'pointnet2_utils.py' and f.name != '__init__.py']
    
    if not model_files:
        # 如果实验目录中没有模型文件，尝试从 logs 目录读取（向后兼容）
        logs_dir = experiment_dir_path / 'logs'
        if logs_dir.exists():
            log_files = list(logs_dir.glob('*.txt'))
            if log_files:
                # 从日志文件名提取模型名（例如：pointnet2_sem_seg_att.txt -> pointnet2_sem_seg_att）
                model_name = log_files[0].stem
            else:
                raise FileNotFoundError(f'在 {experiment_dir} 中找不到模型文件')
        else:
            raise FileNotFoundError(f'在 {experiment_dir} 中找不到模型文件')
    else:
        # 使用找到的第一个模型文件
        model_name = model_files[0].stem
    
    # 将实验目录添加到 sys.path 以便导入模型
    experiment_dir_abs = os.path.abspath(experiment_dir)
    if experiment_dir_abs not in sys.path:
        sys.path.insert(0, experiment_dir_abs)
    
    log_string(f'Loading model: {model_name}')
    MODEL = importlib.import_module(model_name)
    
    # 检查是否是DawNet模型（需要额外参数）
    if 'dawnet' in model_name.lower():
        log_string('检测到DawNet模型，使用额外参数初始化')
        classifier = MODEL.get_model(NUM_CLASSES_MODEL, use_whitening=True, use_attention=True).cuda()
    else:
        classifier = MODEL.get_model(NUM_CLASSES_MODEL).cuda()
    
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    
    # 优先使用 model_state_dict，如果没有则从 G_state_dict 中提取 backbone 权重
    if 'model_state_dict' in checkpoint:
        model_state_dict = checkpoint['model_state_dict']
        log_string('使用 model_state_dict 加载模型')
    elif 'G_state_dict' in checkpoint:
        # 从 G_state_dict 中提取 backbone 的权重（去掉 'backbone.' 前缀）
        G_state_dict = checkpoint['G_state_dict']
        model_state_dict = {}
        for key, value in G_state_dict.items():
            if key.startswith('backbone.'):
                # 去掉 'backbone.' 前缀
                new_key = key[9:]  # len('backbone.') = 9
                model_state_dict[new_key] = value
        log_string('从 G_state_dict 中提取 backbone 权重加载模型')
    else:
        raise KeyError('checkpoint 中既没有 model_state_dict 也没有 G_state_dict')
    
    # 使用 strict=False 允许缺少某些参数
    missing_keys, unexpected_keys = classifier.load_state_dict(model_state_dict, strict=False)
    if missing_keys:
        log_string(f'警告: checkpoint 中缺少以下键: {len(missing_keys)} 个键')
        # 只显示前几个作为示例
        for key in list(missing_keys)[:5]:
            log_string(f'  - {key}')
        if len(missing_keys) > 5:
            log_string(f'  ... 还有 {len(missing_keys) - 5} 个键')
    if unexpected_keys:
        log_string(f'警告: checkpoint 中有意外的键（将被忽略）: {len(unexpected_keys)} 个键')
        # 只显示前几个作为示例
        for key in list(unexpected_keys)[:5]:
            log_string(f'  - {key}')
        if len(unexpected_keys) > 5:
            log_string(f'  ... 还有 {len(unexpected_keys) - 5} 个键')
    
    classifier = classifier.eval()

    # ground_truth = []
    # predictions = []
    # unique = np.arange(NUM_CLASSES)

    with torch.no_grad():
        scene_id = TEST_DATASET_WHOLE_SCENE.file_list
        scene_id = [x[:-4] for x in scene_id]
        num_batches = len(TEST_DATASET_WHOLE_SCENE)

        total_seen_class = [0 for _ in range(NUM_CLASSES_ORIGINAL)]
        total_correct_class = [0 for _ in range(NUM_CLASSES_ORIGINAL)]
        total_iou_deno_class = [0 for _ in range(NUM_CLASSES_ORIGINAL)]

        # 计算被排除类别的索引集合（用于可视化和评估）
        exclude_indices = [class2label[cls] for cls in exclude_classes] if exclude_classes else []
        exclude_indices_set = set(exclude_indices)

        log_string('---- EVALUATION WHOLE SCENE----')

        for batch_idx in range(num_batches):
            print("Inference [%d/%d] %s ..." % (batch_idx + 1, num_batches, scene_id[batch_idx]))
            total_seen_class_tmp = [0 for _ in range(NUM_CLASSES_ORIGINAL)]
            total_correct_class_tmp = [0 for _ in range(NUM_CLASSES_ORIGINAL)]
            total_iou_deno_class_tmp = [0 for _ in range(NUM_CLASSES_ORIGINAL)]
            if args.visual:
                fout = open(os.path.join(visual_dir, scene_id[batch_idx] + '_pred.obj'), 'w')
                fout_gt = open(os.path.join(visual_dir, scene_id[batch_idx] + '_gt.obj'), 'w')

            whole_scene_data = TEST_DATASET_WHOLE_SCENE.scene_points_list[batch_idx]
            whole_scene_label = TEST_DATASET_WHOLE_SCENE.semantic_labels_list[batch_idx]

            # if len(np.intersect1d(np.unique(whole_scene_label), unique)) == 0:
            #     continue
            
            coord, label, idx_data = data_load(whole_scene_data, whole_scene_label)
            pred = torch.zeros((label.size, NUM_CLASSES_MODEL)).cuda()

            idx_size = len(idx_data)
            idx_list, coord_list, feat_list, offset_list  = [], [], [], []
            
            for i in range(idx_size):
                idx_part = idx_data[i]
                coord_part = coord[idx_part]
 
                if args.num_point and coord_part.shape[0] > args.num_point:
                    coord_p, idx_uni, cnt = np.random.rand(coord_part.shape[0]) * 1e-3, np.array([]), 0
                    while idx_uni.size != idx_part.shape[0]:
                        init_idx = np.argmin(coord_p)
                        dist = np.sum(np.power(coord_part - coord_part[init_idx], 2), 1)
                        idx_crop = np.argsort(dist)[:args.num_point]
                        coord_sub, idx_sub = coord_part[idx_crop], idx_part[idx_crop]
                        dist = dist[idx_crop]
                        delta = np.square(1 - dist / np.max(dist))
                        coord_p[idx_crop] += delta
                        coord_sub = input_normalize(coord_sub)
                        idx_list.append(idx_sub), coord_list.append(coord_sub), offset_list.append(idx_sub.size)
                        idx_uni = np.unique(np.concatenate((idx_uni, idx_sub)))
                else:
                    coord_part = input_normalize(coord_part)
                    idx_list.append(idx_part), coord_list.append(coord_part), offset_list.append(idx_part.size)
            batch_num = int(np.ceil(len(idx_list) / args.batch_size))
            for i in range(batch_num):
                s_i, e_i = i * args.batch_size, min((i + 1) * args.batch_size, len(idx_list))
                idx_part, coord_part, feat_part, offset_part = idx_list[s_i:e_i], coord_list[s_i:e_i], feat_list[s_i:e_i], offset_list[s_i:e_i]
                idx_part = np.concatenate(idx_part)
                coord_part = torch.FloatTensor(np.concatenate(coord_part)).cuda(non_blocking=True)
                offset_part = torch.IntTensor(np.cumsum(offset_part)).cuda(non_blocking=True)
                with torch.no_grad():
                    # 检查是否是DawNet模型
                    if 'dawnet' in model_name.lower():
                        # DawNet返回 (seg_pred, domain_pred, feature)，只需要seg_pred
                        pred_part, _, _ = classifier([coord_part, coord_part, offset_part], alpha=0.0)
                    else:
                        pred_part = classifier([coord_part, feat_part, offset_part])[0]  # (n, k)
                torch.cuda.empty_cache()
                pred[idx_part, :] += pred_part

            pred_new_labels = pred.max(1)[1].data.cpu().numpy()
            
            # 将模型预测的新标签空间映射回原始标签空间
            pred_label = np.full(pred_new_labels.shape, -1, dtype=np.int64)
            for new_l in range(len(reverse_mapping)):
                mask = pred_new_labels == new_l
                pred_label[mask] = reverse_mapping[new_l]
            
            # 对于被排除的类别，ground truth中的这些点应该被忽略
            # 创建掩码：只评估保留的类别
            # exclude_indices 已经在循环外计算过了
            eval_mask = np.ones(whole_scene_label.shape[0], dtype=bool)
            if exclude_indices:
                for excl_idx in exclude_indices:
                    eval_mask = eval_mask & (whole_scene_label != excl_idx)
            
            # 只对保留的类别进行评估
            for l in range(NUM_CLASSES_ORIGINAL):
                # 如果这个类别被排除了，跳过评估
                if exclude_classes and classes[l] in exclude_classes:
                    continue
                # 只评估在eval_mask中的点
                mask_l = (whole_scene_label == l) & eval_mask
                mask_pred_l = (pred_label == l) & eval_mask
                total_seen_class_tmp[l] += np.sum(mask_l)
                total_correct_class_tmp[l] += np.sum(mask_pred_l & mask_l)
                total_iou_deno_class_tmp[l] += np.sum(mask_pred_l | mask_l)
                total_seen_class[l] += total_seen_class_tmp[l]
                total_correct_class[l] += total_correct_class_tmp[l]
                total_iou_deno_class[l] += total_iou_deno_class_tmp[l]

            iou_map = np.array(total_correct_class_tmp) / (np.array(total_iou_deno_class_tmp, dtype=np.float32) + 1e-6)
            print(iou_map)
            arr = np.array(total_seen_class_tmp)
            # 只计算保留类别的IoU
            valid_indices = []
            for l in range(NUM_CLASSES_ORIGINAL):
                if exclude_classes and classes[l] in exclude_classes:
                    continue
                if arr[l] > 0:
                    valid_indices.append(l)
            if valid_indices:
                tmp_iou = np.mean(iou_map[valid_indices])
            else:
                tmp_iou = 0.0
            log_string('Mean IoU of %s: %.4f' % (scene_id[batch_idx], tmp_iou))
            print('----------------------------')

            filename = os.path.join(visual_dir, scene_id[batch_idx] + '.txt')
            with open(filename, 'w') as pl_save:
                for i in pred_label:
                    # 如果预测标签为-1（不应该发生），保存0作为默认值
                    label_to_save = int(i) if i >= 0 else 0
                    pl_save.write(str(label_to_save) + '\n')
                pl_save.close()
            
            # data_trans = np.concatenate([whole_scene_data, pred_label[:, np.newaxis], whole_scene_label[:, np.newaxis]], axis=-1)
            # np.save(os.path.join(BASE_DIR, 'data/craslab3d_add',scene_id[batch_idx] + '.npy'), data_trans)
            
            # 在可视化时排除 exclude_classes 中的点
            # exclude_indices_set 已经在循环外计算过了
            for i in range(whole_scene_label.shape[0]):
                # 跳过 easy_view_labels 中的点
                if whole_scene_label[i] in g_easy_view_labels:
                    continue
                # 跳过 exclude_classes 中的点（无论是gt还是pred）
                if whole_scene_label[i] in exclude_indices_set:
                    continue
                # 处理预测标签：如果为-1（不应该发生），使用0作为默认值
                pred_idx = pred_label[i] if pred_label[i] >= 0 else 0
                # 如果预测标签也在排除列表中，跳过（虽然理论上不应该发生，但为了安全）
                if pred_idx in exclude_indices_set:
                    continue
                color = g_label2color[pred_idx]
                color_gt = g_label2color[whole_scene_label[i]]
                if args.visual:
                    fout.write('v %f %f %f %d %d %d\n' % (
                        whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2], color[0], color[1],
                        color[2]))
                    fout_gt.write(
                        'v %f %f %f %d %d %d\n' % (
                            whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2], color_gt[0],
                            color_gt[1], color_gt[2]))
            if args.visual:
                fout.close()
                fout_gt.close()


        # predictions = np.concatenate(predictions)
        # ground_truth = np.concatenate(ground_truth)
        # cm = confusion_matrix(ground_truth, predictions)
        # cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # sns.set(font_scale=1.5)
        # sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes)
        # plt.xlabel('Predicted Labels', fontsize=16)
        # plt.ylabel('True Labels', fontsize=16)
        # plt.title('Confusion Matrix', fontsize=16)

        IoU = np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float32) + 1e-6)
        iou_per_class_str = '------- IoU --------\n'
        
        # 只显示保留类别的IoU
        valid_classes = []
        for l in range(NUM_CLASSES_ORIGINAL):
            # 如果这个类别被排除了，跳过
            if exclude_classes and classes[l] in exclude_classes:
                continue
            if total_seen_class[l] > 0:  # 只显示有数据的类别
                valid_classes.append(l)
                iou_per_class_str += 'class %s, IoU: %.3f \n' % (
                    classes[l] + ' ' * (14+1 - len(classes[l])),
                    total_correct_class[l] / float(total_iou_deno_class[l]))
        
        log_string(iou_per_class_str)
        
        # 只计算保留类别的平均IoU
        if valid_classes:
            valid_ious = IoU[valid_classes]
            log_string('eval point avg class IoU (excluding filtered classes): %f' % np.mean(valid_ious))
            valid_seen = np.array(total_seen_class)[valid_classes]
            valid_correct = np.array(total_correct_class)[valid_classes]
            log_string('eval whole scene point avg class acc: %f' % (
                np.mean(valid_correct / (valid_seen.astype(np.float32) + 1e-6))))
            log_string('eval whole scene point accuracy: %f' % (
                    np.sum(total_correct_class) / float(np.sum(total_seen_class) + 1e-6)))
        else:
            log_string('No valid classes to evaluate')

        print("Done!")


def data_load(points, label):

    coord = points[:, :3]  # N * 3

    idx_data = []

    coord_min = np.min(coord, 0)
    coord -= coord_min
    idx_sort, count = voxelize(coord, args.voxel_size, mode=1)
    for i in range(count.max()):
        idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
        idx_part = idx_sort[idx_select]
        idx_data.append(idx_part)

    return coord, label, idx_data

if __name__ == '__main__':
    args = parse_args()
    main(args)
