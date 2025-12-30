import numpy as np
import random
import torch


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, coord, label):
        for t in self.transforms:
            coord, label = t(coord, label)
        return coord, label


class ToTensor(object):
    def __call__(self, coord, label):
        coord = torch.from_numpy(coord)
        if not isinstance(coord, torch.FloatTensor):
            coord = coord.float()
        label = torch.from_numpy(label)
        if not isinstance(label, torch.LongTensor):
            label = label.long()
        return coord, label


class RandomRotate(object):
    def __init__(self, angle=[0, 0, 1]):
        self.angle = angle

    def __call__(self, coord, label):
        angle_x = np.random.uniform(-self.angle[0], self.angle[0]) * np.pi
        angle_y = np.random.uniform(-self.angle[1], self.angle[1]) * np.pi
        angle_z = np.random.uniform(-self.angle[2], self.angle[2]) * np.pi
        cos_x, sin_x = np.cos(angle_x), np.sin(angle_x)
        cos_y, sin_y = np.cos(angle_y), np.sin(angle_y)
        cos_z, sin_z = np.cos(angle_z), np.sin(angle_z)
        R_x = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])
        R_y = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
        R_z = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]])
        # R = np.dot(R_z, np.dot(R_y, R_x))
        R = np.dot(R_y, R_x)
        coord = np.dot(coord, np.transpose(R))
        return coord, label


class RandomScale(object):
    def __init__(self, scale=[0.9, 1.1], anisotropic=False):
        self.scale = scale
        self.anisotropic = anisotropic

    def __call__(self, coord, label):
        scale = np.random.uniform(self.scale[0], self.scale[1], 3 if self.anisotropic else 1)
        coord *= scale
        return coord, label


class RandomShift(object):
    def __init__(self, shift=[0.2, 0.2, 0]):
        self.shift = shift

    def __call__(self, coord, label):
        shift_x = np.random.uniform(-self.shift[0], self.shift[0])
        shift_y = np.random.uniform(-self.shift[1], self.shift[1])
        shift_z = np.random.uniform(-self.shift[2], self.shift[2])
        coord += [shift_x, shift_y, shift_z]
        return coord, label


class RandomFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, coord, label):
        if np.random.rand() < self.p:
            coord[:, 0] = -coord[:, 0]
        if np.random.rand() < self.p:
            coord[:, 1] = -coord[:, 1]
        return coord, label


class RandomJitter(object):
    def __init__(self, sigma=0.01, clip=0.05):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, coord, label):
        assert (self.clip > 0)
        jitter = np.clip(self.sigma * np.random.randn(coord.shape[0], 3), -1 * self.clip, self.clip)
        coord += jitter
        return coord, label


class ChromaticAutoContrast(object):
    def __init__(self, p=0.2, blend_factor=None):
        self.p = p
        self.blend_factor = blend_factor

    def __call__(self, coord, feat, label):
        if np.random.rand() < self.p:
            lo = np.min(feat, 0, keepdims=True)
            hi = np.max(feat, 0, keepdims=True)
            scale = 255 / (hi - lo)
            contrast_feat = (feat[:, :3] - lo) * scale
            blend_factor = np.random.rand() if self.blend_factor is None else self.blend_factor
            feat[:, :3] = (1 - blend_factor) * feat[:, :3] + blend_factor * contrast_feat
        return coord, feat, label


class ChromaticTranslation(object):
    def __init__(self, p=0.95, ratio=0.05):
        self.p = p
        self.ratio = ratio

    def __call__(self, coord, feat, label):
        if np.random.rand() < self.p:
            tr = (np.random.rand(1, 3) - 0.5) * 255 * 2 * self.ratio
            feat[:, :3] = np.clip(tr + feat[:, :3], 0, 255)
        return coord, feat, label


class ChromaticJitter(object):
    def __init__(self, p=0.95, std=0.005):
        self.p = p
        self.std = std

    def __call__(self, coord, feat, label):
        if np.random.rand() < self.p:
            noise = np.random.randn(feat.shape[0], 3)
            noise *= self.std * 255
            feat[:, :3] = np.clip(noise + feat[:, :3], 0, 255)
        return coord, feat, label


class HueSaturationTranslation(object):
    @staticmethod
    def rgb_to_hsv(rgb):
        # Translated from source of colorsys.rgb_to_hsv
        # r,g,b should be a numpy arrays with values between 0 and 255
        # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
        rgb = rgb.astype('float')
        hsv = np.zeros_like(rgb)
        # in case an RGBA array was passed, just copy the A channel
        hsv[..., 3:] = rgb[..., 3:]
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maxc = np.max(rgb[..., :3], axis=-1)
        minc = np.min(rgb[..., :3], axis=-1)
        hsv[..., 2] = maxc
        mask = maxc != minc
        hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
        rc = np.zeros_like(r)
        gc = np.zeros_like(g)
        bc = np.zeros_like(b)
        rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
        gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
        bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
        hsv[..., 0] = np.select([r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
        hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
        return hsv

    @staticmethod
    def hsv_to_rgb(hsv):
        # Translated from source of colorsys.hsv_to_rgb
        # h,s should be a numpy arrays with values between 0.0 and 1.0
        # v should be a numpy array with values between 0.0 and 255.0
        # hsv_to_rgb returns an array of uints between 0 and 255.
        rgb = np.empty_like(hsv)
        rgb[..., 3:] = hsv[..., 3:]
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        i = (h * 6.0).astype('uint8')
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
        rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
        rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
        rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
        return rgb.astype('uint8')

    def __init__(self, hue_max=0.5, saturation_max=0.2):
        self.hue_max = hue_max
        self.saturation_max = saturation_max

    def __call__(self, coord, feat, label):
        # Assume feat[:, :3] is rgb
        hsv = HueSaturationTranslation.rgb_to_hsv(feat[:, :3])
        hue_val = (np.random.rand() - 0.5) * 2 * self.hue_max
        sat_ratio = 1 + (np.random.rand() - 0.5) * 2 * self.saturation_max
        hsv[..., 0] = np.remainder(hue_val + hsv[..., 0] + 1, 1)
        hsv[..., 1] = np.clip(sat_ratio * hsv[..., 1], 0, 1)
        feat[:, :3] = np.clip(HueSaturationTranslation.hsv_to_rgb(hsv), 0, 255)
        return coord, feat, label


class RandomDropColor(object):
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, coord, feat, label):
        if np.random.rand() < self.p:
            feat[:, :3] = 0
            # feat[:, :3] = 127.5
        return coord, feat, label

# ==== TAIL-AWARE CUBOID MIXING ====
def tacm(param, split_sampler, class_names, pc1, pc2):
    # ===== pre-process
    xyz_middle, label = pc1
    xyz_middle2, label2 = pc2
    xyz_middle -= (xyz_middle.min(0) + xyz_middle.max(0)) / 2.0
    xyz_middle2 -= (xyz_middle2.min(0) + xyz_middle2.max(0)) / 2.0

    # ===== cuboid split
    split_coord_xyz, split_range = split_space(xyz_middle, param.split)
    split_coord_xyz2, split_range2 = split_space(xyz_middle2, param.split)
    split_idx, split_info = get_split_idx(
        param, xyz_middle, label, split_coord_xyz, split_range, calc_split=True, n_classes=len(class_names)
    )
    split_idx2, _ = get_split_idx(param, xyz_middle2, label2, split_coord_xyz2, split_range2)
    total_splits = param.split[0] * param.split[1] * param.split[2]
    split_status = split_info['split_status']

    # ===== cuboid mixing, 1: source, 0: target, e.g. [0,1,1,0]
    concat = check_p(param)
    if concat:
        concat_seq = (np.random.rand(total_splits) < param.mix_ratio).astype(np.uint8)
    else:
        concat_seq = np.array([0] * total_splits)

    # ===== cuboid permutation. e.g. [2,1,4,3]
    permute = check_p(param.permute_cuboid)
    if permute:
        # target
        split_idx, permuted_cuboid_coord_xyz, _, split_others = permute_cuboid(
            param.permute_cuboid, int(total_splits - concat_seq.sum()), split_idx, split_coord_xyz, split_range,
            xyz=xyz_middle, label=label, split_status=split_status
        )
        split_status = split_others['split_status']
        # source
        split_idx2, permuted_cuboid_coord_xyz2, _, _ = permute_cuboid(
            param.permute_cuboid, int(concat_seq.sum()), split_idx2, split_coord_xyz2, split_range2,
            xyz=xyz_middle2, label=label2
        )
        permuted_cuboid_coord_xyzs = [permuted_cuboid_coord_xyz, permuted_cuboid_coord_xyz2]
    else:
        tar_mapper = np.where(concat_seq == 0, np.cumsum(concat_seq == 0), 0) - 1
        split_idx = tar_mapper[split_idx]
        split_status = split_status[concat_seq == 0]
        src_mapper = np.where(concat_seq == 1, np.cumsum(concat_seq == 1), 0) - 1
        split_idx2 = src_mapper[split_idx2]

    # ===== get target tail-aware cuboids
    tail_cuboids = tail_cuboids_from_sampler(
        param, int(total_splits - concat_seq.sum()), split_status, split_sampler, label=label
    )

    split_idxs = [split_idx, split_idx2]
    split_coord_xyzs = [split_coord_xyz, split_coord_xyz2]
    split_ranges = [split_range, split_range2]
    xyz_middles = [xyz_middle, xyz_middle2]
    masks = [np.zeros(xyz_middle.shape[0], dtype=bool), np.zeros(xyz_middle2.shape[0], dtype=bool)]
    
    # ===== get mixed cuboids
    new_split_coords = []
    new_split_range = []

    concat_seq_tar = concat_seq[concat_seq == 0]
    for i in range(len(tail_cuboids)):
        concat_seq_tar[-i - 1] = 2
    concat_seq[concat_seq == 0] = concat_seq_tar

    ptrs = [0, 0, 0]
    for s in range(total_splits):
        domain = concat_seq[s]
        if domain == 2:  # from tail-aware cuboid sampler
            _split = tail_cuboids[ptrs[domain]]
            _split[..., 0:3] += split_coord_xyzs[0][s] - _split[..., 0:3].max(0)  # use the target domain split_coord
            _split[..., 0:3] = transform_xyz(_split[..., 0:3], param)
            ptrs[domain] += 1
            new_split_coords.append(split_coord_xyzs[0][s])
            new_split_range.append(split_ranges[0][s])
        else:
            xyz_idx_s = split_idxs[domain] == ptrs[domain]  # the head of the queue
            if permute:
                xyz_middles[domain][xyz_idx_s] += split_coord_xyzs[domain][s] - permuted_cuboid_coord_xyzs[domain][ptrs[domain]]
            xyz_middles[domain][xyz_idx_s] = transform_xyz(xyz_middles[domain][xyz_idx_s], param)
            masks[domain][xyz_idx_s] = True
            ptrs[domain] += 1
            new_split_coords.append(split_coord_xyzs[domain][s])
            new_split_range.append(split_ranges[domain][s])

    xyz_middle, label = filter_by_index([xyz_middles[0], label], masks[0])
    xyz_middle2, label2 = filter_by_index([xyz_middles[1], label2], masks[1])


    if len(tail_cuboids) > 0:
        tail_cuboids = np.concatenate(tail_cuboids, axis=0)
    else:
        tail_cuboids = np.random.rand(0, 4).astype(xyz_middle.dtype)
    xyz_middle = np.concatenate((xyz_middle, xyz_middle2, tail_cuboids[..., 0:3]), axis=0)
    xyz_middle -= xyz_middle.mean(0)
    label = np.concatenate((label, label2, tail_cuboids[..., 3]), axis=0)
    others_merged = {}

    others_merged['pc1_mask'] = np.where(np.arange(label.shape[0]) < masks[0].sum(), True, False)
    others_merged['pc2_mask'] = ~others_merged['pc1_mask']
    others_merged['tar_tail_splits'] = split_info['tail_splits']
    if param.cuboid_queue.enabled:
        others_merged['tar_splits_class_ratio'] = \
            np.histogram(tail_cuboids[..., 3], bins=np.arange(len(class_names) + 1))[0][split_sampler.tail_class_idx]
    else:
        others_merged['tar_splits_class_ratio'] = np.zeros(3)
    return xyz_middle, label, others_merged

# random split
def split_space(xyz, split):
    assert len(split) == 3  # 3 dimensions, [2, 1, 1]
    xyz_min, xyz_max = xyz.min(0), xyz.max(0)
    xyz_range = xyz_max - xyz_min + 0.001
    split_ratio = (1.0 / np.array(split, dtype=np.float32).reshape(3, 1)).tolist()
    split_ratio = [np.cumsum(r * split[i]) for (i, r) in enumerate(split_ratio)]  # [[0.5,1],[1],[1]]
    split_ratio = [np.append(r[:-1] + (np.random.rand() - 0.5) * 0.2, 1.0) for r in split_ratio]  # [[0.44,1],[1],[1]]
    split_ratio_range = [np.append(r[0], np.array(r[1:]) - np.array(r[:-1])) for r in split_ratio]
    total_splits = split[0] * split[1] * split[2]
    split_coord = np.array([
        [split_ratio[0][i // (split[1] * split[2])] * xyz_range[0] + xyz_min[0],
         split_ratio[1][i % (split[1] * split[2]) // split[2]] * xyz_range[1] + xyz_min[1],
         split_ratio[2][i % (split[2])] * xyz_range[2] + xyz_min[2]]
        for i in range(total_splits)])  # (total_splits, 3), [[0.5, 0.4, 0.3], [1.0, 0.4, 0.3]]
    split_range = np.array([
        [split_ratio_range[0][i // (split[1] * split[2])] * xyz_range[0],
         split_ratio_range[1][i % (split[1] * split[2]) // split[2]] * xyz_range[1],
         split_ratio_range[2][i % (split[2])] * xyz_range[2]]
        for i in range(total_splits)])  # (total_splits, 3),
    return split_coord, split_range

def get_split_idx(param, xyz, label, split_coord_xyz, split_range, **args):
    split_idx = np.full(xyz.shape[0], 255, dtype=np.int8)
    tail_splits = [[] for _ in range(param.cuboid_queue.num_class)]
    split_status = []
    for s in range(split_coord_xyz.shape[0]):
        xyz_idx_s = xyz_idx_in_split(xyz, split_coord_xyz[s], split_range[s])
        split_idx[xyz_idx_s] = s
        if param.cuboid_queue.enabled and 'calc_split' in args and xyz_idx_s.sum() > 0 and label[xyz_idx_s].min() < 255:
            labels_s = label[xyz_idx_s]
            # Only count valid classes [0, n_classes-1], avoid ignored classes/out-of-bounds causing all zeros and density division by zero
            if 'n_classes' in args and args['n_classes'] is not None:
                valid_mask = (labels_s >= 0) & (labels_s < args['n_classes'])
                labels_s = labels_s[valid_mask]
                bins = np.arange(args['n_classes'] + 1)
            else:
                bins = np.arange(labels_s.max() + 2)
            hist = np.histogram(labels_s, bins=bins)[0]
            total = hist.sum()
            class_ratio = (hist / total) if total > 0 else np.zeros(bins.shape[0] - 1, dtype=np.float32)
            status = (class_ratio > param.cuboid_queue.class_ratio)[param.cuboid_queue.tail_class_idx]
            split_status.append(np.any(status))
            for i in range(param.cuboid_queue.num_class):
                if status[i]:
                    tail_splits[i].append(np.concatenate((xyz[xyz_idx_s], label[xyz_idx_s].reshape(-1, 1)), axis=-1))
        else:
            split_status.append(False)
    return split_idx, {'tail_splits': tail_splits, 'split_status': np.array(split_status)}

def xyz_idx_in_split(xyz, split_max, range):
    return np.all(xyz < split_max, axis=-1) & np.all(xyz >= split_max - range, axis=-1)

def check_p(key):
    return (not isinstance(key, dict)) or ('p' not in key) or (np.random.rand() < key['p'])

def tail_cuboids_from_sampler(param, n, split_status, split_sampler, **args):
    # replace original splits with splits in cuboid_queue
    supp_queues = []
    if param.cuboid_queue.enabled:
        # split_status = get_split_status(param.cuboid_queue.class_thres, args['label'], n, split_idx)
        n_eligible_splits = param.cuboid_queue.num_cuboid
        n_eligible_splits = int((n_eligible_splits // 1) + int(np.random.rand() < n_eligible_splits % 1))
        n_curr_splits = split_status.sum()
        supp_num = min(n, n_eligible_splits) - n_curr_splits
        if supp_num > 0:
            supp_queues = split_sampler.get_split(supp_num)
    return supp_queues

def permute_cuboid(param, n, split_idx, split_coord_xyz, split_range, **args):
    n_split = split_coord_xyz.shape[0]
    permuted_idx = np.random.permutation(np.arange(n_split))
    permuted_cuboid_idx = np.argsort(permuted_idx)[split_idx]
    permuted_cuboid_coord_xyz = split_coord_xyz[permuted_idx][:n]
    permuted_cuboid_range = split_range[permuted_idx][:n]

    if 'split_status' in args:
        args['split_status'] = args['split_status'][permuted_idx][:n]

    return permuted_cuboid_idx, permuted_cuboid_coord_xyz, permuted_cuboid_range, args

def transform_xyz(xyz, param):
    if xyz.shape[0] > 0:
        mv_direc = - xyz.mean(0)
        xyz += mv_direc * 0.1
    return xyz

def filter_by_index(e_list, idx):
    filtered_e_list = list()
    for e in e_list:
        filtered_e_list.append(e[idx])
    return filtered_e_list