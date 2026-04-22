import os
import glob
import numpy as np
from torch.utils.data import Dataset


def load_img_name_list(dataset_path):
    """从文件中加载图像名称列表"""
    img_gt_name_list = open(dataset_path).readlines()
    img_name_list = [img_gt_name.strip() for img_gt_name in img_gt_name_list]
    return img_name_list


def load_image_label_list_from_npy(img_name_list, label_file_path=None):
    """从npy文件中加载图像标签列表"""
    if label_file_path is None:
        label_file_path = "voc12/cls_labels.npy"
    
    cls_labels_dict = np.load(label_file_path, allow_pickle=True).item()
    label_list = []
    for id in img_name_list:
        if id not in cls_labels_dict.keys():
            img_name = id + ".jpg"
        else:
            img_name = id
        label_list.append(cls_labels_dict[img_name])
    return label_list


class VOC12Dataset(Dataset):
    """VOC12数据集类，用于DINO超像素生成"""
    
    def __init__(
        self,
        img_name_list_path,
        voc12_root,
        train=True,
        gen_attn=False,
        transform=None,
    ):
        """
        初始化VOC12数据集
        
        Args:
            img_name_list_path: 包含图像名称列表文件的目录路径
            voc12_root: VOC2012数据集根目录
            train: 是否为训练模式
            gen_attn: 是否生成注意力图
            transform: 图像变换
        """
        # 构建图像名称列表文件路径
        if os.path.isfile(img_name_list_path):
            # 如果传入的是文件路径，直接使用
            img_name_list_file = img_name_list_path
        else:
            # 如果传入的是目录路径，构建文件名
            img_name_list_file = os.path.join(
                img_name_list_path, f'{"train_aug" if train or gen_attn else "val"}_id.txt'
            )
        
        self.img_name_list = load_img_name_list(img_name_list_file)
        
        # 加载标签文件
        label_file_path = os.path.join(
            os.path.dirname(img_name_list_file), "cls_labels.npy"
        )
        
        # 如果标签文件存在，则加载标签
        if os.path.exists(label_file_path):
            self.label_list = load_image_label_list_from_npy(
                self.img_name_list, label_file_path
            )
        else:
            # 如果标签文件不存在，创建虚拟标签（全零）
            self.label_list = [np.zeros(20) for _ in self.img_name_list]
        
        self.voc12_root = voc12_root
        self.transform = transform
    
    def __getitem__(self, idx):
        """获取数据集中的一个样本"""
        img_name = self.img_name_list[idx]
        
        # 构建图像文件路径
        img_path = os.path.join(self.voc12_root, "JPEGImages", f"{img_name}.jpg")
        
        # 返回图像路径和标签
        return {
            'img_name': img_name,
            'img_path': img_path,
            'label': self.label_list[idx] if hasattr(self, 'label_list') else np.zeros(20)
        }
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.img_name_list)


class COCODataset(Dataset):
    """COCO数据集类，用于DINO超像素生成"""
    
    def __init__(
        self,
        coco_root,
        split="train",
        transform=None,
    ):
        """
        初始化COCO数据集
        
        Args:
            coco_root: COCO数据集根目录（包含train2014/和val2014/子目录）
            split: 数据集分割，可选 "train" 或 "val"（默认: "train"）
            transform: 图像变换
        """
        self.coco_root = coco_root
        self.split = split
        self.transform = transform
        
        # 确定图像目录
        if split == "train":
            self.img_dir = os.path.join(coco_root, "train2014")
        elif split == "val":
            self.img_dir = os.path.join(coco_root, "val2014")
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'val'")
        
        # 检查目录是否存在
        if not os.path.exists(self.img_dir):
            raise ValueError(f"Image directory not found: {self.img_dir}")
        
        # 扫描图像文件（支持 .jpg 格式）
        img_pattern = os.path.join(self.img_dir, "*.jpg")
        img_files = sorted(glob.glob(img_pattern))
        
        if len(img_files) == 0:
            raise ValueError(f"No images found in {self.img_dir}")
        
        # 提取图像名称（不含路径和扩展名）
        self.img_name_list = []
        for img_file in img_files:
            img_name = os.path.splitext(os.path.basename(img_file))[0]
            self.img_name_list.append(img_name)
        
        # COCO 数据集不需要标签文件（超像素生成不需要类别标签）
        self.label_list = None
    
    def __getitem__(self, idx):
        """获取数据集中的一个样本"""
        img_name = self.img_name_list[idx]
        
        # 构建图像文件路径
        img_path = os.path.join(self.img_dir, f"{img_name}.jpg")
        
        # 返回图像路径和图像名称
        return {
            'img_name': img_name,
            'img_path': img_path,
            'label': np.zeros(1)  # 占位符，实际不使用
        }
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.img_name_list)


def get_spix_label(img_np: np.ndarray):
    # 定义常量
    N_COMPONENTS = 196
    COMPACTNESS = 10

    try:
        # 使用 fast_slic 生成超像素
        slic = SlicAvx2(num_components=N_COMPONENTS, compactness=COMPACTNESS)
        sp_label = slic.iterate(img_np)  # np.ndarray
        
        # 验证和修正superpixel标签
        # 确保标签值在有效范围内 [0, N_COMPONENTS-1]
        max_label = sp_label.max()
        min_label = sp_label.min()
        
        # 如果标签超出范围，进行重新映射
        if max_label >= N_COMPONENTS or min_label < 0:
            unique_labels = np.unique(sp_label)
            # 创建映射字典，将标签重新映射到 [0, min(len(unique_labels)-1, N_COMPONENTS-1)]
            max_allowed = min(len(unique_labels) - 1, N_COMPONENTS - 1)
            label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels[:max_allowed + 1])}
            
            # 对于超出范围的标签，映射到最后一个有效标签
            for label in unique_labels[max_allowed + 1:]:
                label_map[label] = max_allowed
            
            # 应用映射
            sp_label_fixed = np.zeros_like(sp_label, dtype=np.int32)
            for old_label, new_label in label_map.items():
                sp_label_fixed[sp_label == old_label] = new_label
            sp_label = sp_label_fixed
        
        # 确保数据类型正确
        sp_label = sp_label.astype(np.int32)
        
        # 最终验证
        assert sp_label.min() >= 0, f"Superpixel labels contain negative values: {sp_label.min()}"
        assert sp_label.max() < N_COMPONENTS, f"Superpixel labels exceed maximum: {sp_label.max()} >= {N_COMPONENTS}"
        
        return sp_label
        
    except Exception as e:
        print(f"Error in get_spix_label: {e}")
        print(f"Image shape: {img_np.shape}, dtype: {img_np.dtype}")
        print(f"Image min/max: {img_np.min()}/{img_np.max()}")
        # 降级到简单的网格标签作为备选方案
        h, w = img_np.shape[:2]
        grid_h = int(np.sqrt(N_COMPONENTS * h / w))
        grid_w = N_COMPONENTS // grid_h
        sp_label = np.zeros((h, w), dtype=np.int32)
        for i in range(grid_h):
            for j in range(grid_w):
                y_start = i * h // grid_h
                y_end = (i + 1) * h // grid_h
                x_start = j * w // grid_w
                x_end = (j + 1) * w // grid_w
                sp_label[y_start:y_end, x_start:x_end] = i * grid_w + j
        return sp_label
