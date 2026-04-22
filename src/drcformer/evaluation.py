import os
import pandas as pd
import numpy as np
from PIL import Image
import multiprocessing
import argparse
import time
from tqdm import tqdm

# VOC2012数据集的21个类别，包括背景类
VOC_CATEGORIES = ['background','aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow',
                  'diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']

COCO_CATEGORIES = [
    'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# COCO数据集的80个类别ID（不连续，范围1-90）
# 缺失的ID：12, 26, 29, 30, 45, 66, 68, 69, 71, 83
COCO_CATEGORY_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
    41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
    59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 84, 85, 86, 87, 88, 89, 90
]

# 创建COCO原始ID到连续索引的映射
# COCO原始ID (1-90) -> 连续索引 (1-80)，索引0留给背景类
COCO_ID_TO_IDX = {coco_id: idx + 1 for idx, coco_id in enumerate(COCO_CATEGORY_IDS)}
COCO_ID_TO_IDX[0] = 0  # 背景类映射到索引0

# 创建连续索引到COCO原始ID的反向映射
IDX_TO_COCO_ID = {idx: coco_id for coco_id, idx in COCO_ID_TO_IDX.items()}

def do_python_eval(predict_folder, gt_folder, name_list, num_cls=21, input_type='png', threshold=1.0, printlog=False, num_workers=64):
    """
    语义分割评估的核心函数，计算IoU、mIoU、FP、FN等指标
    
    Args:
        predict_folder: 预测结果文件夹路径
        gt_folder: 真实标注文件夹路径  
        name_list: 图像名称列表
        num_cls: 类别数量，默认21（VOC2012）
        input_type: 输入类型，'png'或'npy'
        threshold: 背景阈值（仅对npy格式有效）
        printlog: 是否打印详细日志
        num_workers: 并行评估的进程数，默认64
    
    Returns:
        loglist: 包含各类别IoU、mIoU、FP、FN的字典
    """
    # 检查预测结果目录是否存在
    if not os.path.exists(predict_folder):
        print(f"Error: Predict folder not found: {predict_folder}")
        return {'mIoU': 0.0, 'FP': 0.0, 'FN': 0.0}
    
    # 检查预测目录中是否有匹配的预测文件
    file_ext = '.png' if input_type == 'png' else '.npy'
    found_files = []
    for name in name_list[:min(100, len(name_list))]:  # 只检查前100个文件作为样本
        predict_file = os.path.join(predict_folder, f'{name}{file_ext}')
        if os.path.exists(predict_file):
            found_files.append(name)
    
    if len(found_files) == 0:
        # 检查目录中是否有任何匹配格式的文件
        all_files = os.listdir(predict_folder) if os.path.isdir(predict_folder) else []
        matching_files = [f for f in all_files if f.endswith(file_ext)]
        
        error_msg = f"\n{'='*60}\n"
        error_msg += f"错误：在预测目录中未找到任何预测文件\n"
        error_msg += f"{'='*60}\n"
        error_msg += f"预测目录: {predict_folder}\n"
        error_msg += f"期望的文件格式: {file_ext}\n"
        error_msg += f"检查的文件数量: {min(100, len(name_list))}\n"
        if matching_files:
            error_msg += f"目录中找到的 {file_ext} 文件数量: {len(matching_files)}\n"
            error_msg += f"示例文件: {matching_files[0] if matching_files else 'N/A'}\n"
        else:
            error_msg += f"目录中未找到任何 {file_ext} 文件\n"
        error_msg += f"\n可能的原因：\n"
        if input_type == 'png':
            error_msg += f"  1. PSA 推理步骤未保存 PNG 文件（可能使用了 --no-save-pseudo-png）\n"
            error_msg += f"  2. 解决方案：重新运行 PSA 推理步骤，确保未使用 --no-save-pseudo-png 参数\n"
            error_msg += f"     例如：python src/drcformer/run_psa.py --steps 3\n"
        else:
            error_msg += f"  1. CAM 生成步骤未完成或文件路径不正确\n"
        error_msg += f"{'='*60}\n"
        print(error_msg)
        return {'mIoU': 0.0, 'FP': 0.0, 'FN': 0.0, 'processed_files': 0}
    
    # 初始化多进程共享变量，用于统计TP、P、T值
    # TP: True Positive（真正例）
    # P: Predicted Positive（预测正例总数）
    # T: True Positive（真实正例总数）
    TP = []
    P = []
    T = []
    for i in range(num_cls):
        TP.append(multiprocessing.Value('i', 0, lock=True))  # 每个类别的TP值
        P.append(multiprocessing.Value('i', 0, lock=True))   # 每个类别的预测像素总数
        T.append(multiprocessing.Value('i', 0, lock=True))   # 每个类别的真实像素总数
    
    # 初始化进度计数器
    processed_count = multiprocessing.Value('i', 0, lock=True)
    # 初始化成功处理的文件计数器
    successful_files = multiprocessing.Value('i', 0, lock=True)
    # 初始化跳过的文件计数器
    skipped_files = multiprocessing.Value('i', 0, lock=True)

    def compare(start, step, TP, P, T, input_type, threshold, predict_folder, gt_folder, name_list, num_cls, processed_count, successful_files, skipped_files):
        """
        多进程处理函数，负责比较预测结果和真实标注
        
        Args:
            start: 起始索引
            step: 步长（进程数）
            TP, P, T: 多进程共享的统计变量
            input_type: 输入文件类型
            threshold: 背景阈值
            predict_folder: 预测结果文件夹路径
            gt_folder: 真实标注文件夹路径
            name_list: 图像名称列表
            num_cls: 类别数量
            processed_count: 已处理图像数量的共享计数器
        """
        for idx in range(start,len(name_list),step):
            name = name_list[idx]
            
            # 根据输入类型加载预测结果
            if input_type == 'png':
                # PNG格式：直接加载分割掩码图像
                predict_file = os.path.join(predict_folder,'%s.png'%name)
                if not os.path.exists(predict_file):
                    skipped_files.acquire()
                    skipped_files.value += 1
                    skipped_files.release()
                    continue
                predict = np.array(Image.open(predict_file))
                
                # COCO数据集：将原始类别ID转换为连续索引（0-80）
                if num_cls == 81:
                    predict_converted = np.zeros_like(predict)
                    for coco_id, idx in COCO_ID_TO_IDX.items():
                        predict_converted[predict == coco_id] = idx
                    predict = predict_converted
            elif input_type == 'npy':
                # NPY格式：加载类别激活图字典，转换为分割掩码
                predict_file = os.path.join(predict_folder,'%s.npy'%name)
                if not os.path.exists(predict_file):
                    skipped_files.acquire()
                    skipped_files.value += 1
                    skipped_files.release()
                    continue
                predict_dict = np.load(predict_file, allow_pickle=True).item()
                h, w = list(predict_dict.values())[0].shape
                
                # 构建多通道张量，每个通道对应一个类别的激活图
                tensor = np.zeros((num_cls,h,w),np.float32)
                
                if num_cls == 81:
                    # COCO数据集：keys是COCO原始类别ID（1-90之间的不连续值）
                    # 需要转换为连续索引（0-80）
                    for coco_id, cam in predict_dict.items():
                        if coco_id in COCO_ID_TO_IDX:
                            idx = COCO_ID_TO_IDX[coco_id]
                            tensor[idx] = cam
                else:
                    # VOC数据集：检测文件格式
                    # - 原始CAM文件：key = 0-19（前景类索引），需要 +1 偏移
                    # - CRF export文件：key = 0-20（0=背景，1-20=前景），不需要偏移
                    # 通过检查 key=0 是否存在且 max(keys) 是否 >= num_cls-1 来判断
                    keys = list(predict_dict.keys())
                    max_key = max(keys) if keys else 0
                    has_bg_channel = 0 in keys and max_key >= num_cls - 1
                    
                    for key in predict_dict.keys():
                        if has_bg_channel:
                            # CRF export 格式：key 已经是 0-20（0=背景，1-20=前景）
                            if key < num_cls:
                                tensor[key] = predict_dict[key]
                        else:
                            # 原始 CAM 格式：key 是 0-19（前景类），需要 +1
                            if key + 1 < num_cls:
                                tensor[key+1] = predict_dict[key]
                    
                    # VOC数据集：设置背景通道的阈值（仅当背景通道未被设置时）
                    if not has_bg_channel:
                        tensor[0,:,:] = threshold
                
                # COCO数据集：设置背景通道阈值
                if num_cls == 81:
                    tensor[0,:,:] = threshold
                
                # 通过argmax获得最终的分割掩码
                predict = np.argmax(tensor, axis=0).astype(np.uint8)

            # 加载真实标注
            gt_file = os.path.join(gt_folder,'%s.png'%name)
            gt = np.array(Image.open(gt_file))
            
            # COCO数据集：将原始类别ID转换为连续索引（0-80）
            if num_cls == 81:
                gt_converted = np.full_like(gt, 255)  # 默认为ignore区域（255）
                for coco_id, idx in COCO_ID_TO_IDX.items():
                    gt_converted[gt == coco_id] = idx
                gt_converted[gt == 255] = 255  # 保留原有的ignore区域
                gt = gt_converted
            
            # 如果预测结果和真实标注尺寸不匹配，调整真实标注尺寸
            if gt.shape != predict.shape:
                gt_pil = Image.fromarray(gt)
                gt_pil = gt_pil.resize((predict.shape[1], predict.shape[0]), Image.NEAREST)
                gt = np.array(gt_pil)
            
            # 创建有效像素掩码（排除标注值为255的忽略区域）
            cal = gt<255
            
            # 创建正确预测的掩码（预测值等于真实值且在有效区域内）
            mask = (predict==gt) * cal
      
            # 统计每个类别的TP、P、T值
            for i in range(num_cls):
                # 统计预测为类别i的像素数（在有效区域内）
                P[i].acquire()
                P[i].value += np.sum((predict==i)*cal)
                P[i].release()
                
                # 统计真实标注为类别i的像素数（在有效区域内）
                T[i].acquire()
                T[i].value += np.sum((gt==i)*cal)
                T[i].release()
                
                # 统计正确预测为类别i的像素数（真正例）
                TP[i].acquire()
                TP[i].value += np.sum((gt==i)*mask)
                TP[i].release()
            
            # 更新进度计数器
            processed_count.acquire()
            processed_count.value += 1
            processed_count.release()
            
            # 更新成功处理的文件计数器
            successful_files.acquire()
            successful_files.value += 1
            successful_files.release()
    # 启动指定数量的进程并行处理图像比较
    p_list = []
    for i in range(num_workers):
        p = multiprocessing.Process(target=compare, args=(i, num_workers, TP, P, T, input_type, threshold, predict_folder, gt_folder, name_list, num_cls, processed_count, successful_files, skipped_files))
        p.start()
        p_list.append(p)
    
    # 使用进度条监控处理进度
    # 当 threshold 为 None 时（如 png 格式），不显示阈值信息
    threshold_str = f"阈值={threshold:.2f}" if threshold is not None else "阈值=N/A"
    with tqdm(total=len(name_list), desc=f"评估中 ({threshold_str})", unit="img") as pbar:
        while any(p.is_alive() for p in p_list):
            current = processed_count.value
            pbar.n = current
            pbar.refresh()
            time.sleep(0.1)
        # 最终更新进度条
        pbar.n = processed_count.value
        pbar.refresh()
    
    # 等待所有进程完成
    for p in p_list:
        p.join()
    
    # 检查处理结果
    total_processed = successful_files.value
    total_skipped = skipped_files.value
    
    if total_processed == 0:
        print(f"\n{'='*60}")
        print(f"警告：没有成功处理任何文件！")
        print(f"{'='*60}")
        print(f"总文件数: {len(name_list)}")
        print(f"成功处理: {total_processed}")
        print(f"跳过文件: {total_skipped}")
        print(f"{'='*60}\n")
        return {'mIoU': 0.0, 'FP': 0.0, 'FN': 0.0, 'processed_files': 0, 'skipped_files': total_skipped}
    
    if total_skipped > 0 and printlog:
        print(f"\n注意：跳过了 {total_skipped} 个文件（文件不存在）")
        print(f"成功处理了 {total_processed} 个文件\n")
    
    # 计算各种评估指标
    IoU = []      # 交并比
    T_TP = []     # 召回率相关
    P_TP = []     # 精确率相关
    FP_ALL = []   # 假正例率
    FN_ALL = []   # 假负例率
    
    for i in range(num_cls):
        # IoU = TP / (TP + FP + FN) = TP / (T + P - TP)
        IoU.append(TP[i].value/(T[i].value+P[i].value-TP[i].value+1e-10))
        
        # T_TP = T / TP（真实正例与真正例的比值）
        T_TP.append(T[i].value/(TP[i].value+1e-10))
        
        # P_TP = P / TP（预测正例与真正例的比值）
        P_TP.append(P[i].value/(TP[i].value+1e-10))
        
        # FP率 = FP / (TP + FP + FN) = (P - TP) / (T + P - TP)
        FP_ALL.append((P[i].value-TP[i].value)/(T[i].value + P[i].value - TP[i].value + 1e-10))
        
        # FN率 = FN / (TP + FP + FN) = (T - TP) / (T + P - TP)
        FN_ALL.append((T[i].value-TP[i].value)/(T[i].value + P[i].value - TP[i].value + 1e-10))
    # 选择合适的类别名称列表
    if num_cls == 21:
        current_categories = VOC_CATEGORIES
    elif num_cls == 81:
        current_categories = COCO_CATEGORIES
    else:
        current_categories = ['class_%d' % i for i in range(num_cls)]
        
    # 构建结果字典
    loglist = {}
    
    # 记录每个类别的IoU（转换为百分比）
    for i in range(num_cls):
        if i < len(current_categories):
            loglist[current_categories[i]] = IoU[i] * 100
        else:
            loglist['class_%d' % i] = IoU[i] * 100
    
    # 计算平均IoU（mIoU）
    miou = np.mean(np.array(IoU))
    loglist['mIoU'] = miou * 100
    
    # 计算平均假正例率
    fp = np.mean(np.array(FP_ALL))
    loglist['FP'] = fp * 100
    
    # 计算平均假负例率
    fn = np.mean(np.array(FN_ALL))
    loglist['FN'] = fn * 100
    
    # 添加处理统计信息
    loglist['processed_files'] = total_processed
    loglist['skipped_files'] = total_skipped
    
    # 打印详细的评估结果
    if printlog:
        # 按两列格式打印每个类别的IoU
        for i in range(num_cls):
            cat_name = current_categories[i] if i < len(current_categories) else 'class_%d' % i
            if i%2 != 1:  # 偶数索引：同行打印，不换行
                print('%11s:%7.3f%%'%(cat_name,IoU[i]*100),end='\t')
            else:         # 奇数索引：同行打印后换行
                print('%11s:%7.3f%%'%(cat_name,IoU[i]*100))
        
        # 打印分隔线和总体指标
        print('\n======================================================')
        print('%11s:%7.3f%%'%('mIoU',miou*100))  # 平均IoU
        print('\n')
        print(f'FP = {fp*100}, FN = {fn*100}')      # 假正例率和假负例率
    return loglist

def writedict(file, dictionary):
    """
    将字典内容格式化写入文件
    
    Args:
        file: 文件对象
        dictionary: 要写入的字典数据
    """
    s = ''
    for key in dictionary.keys():
        sub = '%s:%s  '%(key, dictionary[key])
        s += sub
    s += '\n'
    file.write(s)

def writelog(filepath, metric, comment):
    """
    将评估结果写入日志文件
    
    Args:
        filepath: 日志文件路径
        metric: 评估指标字典
        comment: 评估注释信息
    """
    filepath = filepath
    logfile = open(filepath,'a')
    import time
    # 写入时间戳
    logfile.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    # 写入评估注释
    logfile.write('\t%s\n'%comment)
    # 写入评估指标
    writedict(logfile, metric)
    # 写入分隔线
    logfile.write('=====================================\n')
    logfile.close()


if __name__ == '__main__':
    # 命令行参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", default='./VOC2012/ImageSets/Segmentation/train.txt', type=str, 
                       help="图像名称列表文件路径")
    parser.add_argument("--predict_dir", default='./out_rw', type=str, 
                       help="预测结果文件夹路径")
    parser.add_argument("--gt_dir", default='./VOC2012/SegmentationClass', type=str, 
                       help="真实标注文件夹路径")
    parser.add_argument('--logfile', default='./evallog.txt',type=str, 
                       help="日志文件路径")
    parser.add_argument('--comment', required=True, type=str, 
                       help="评估注释信息")
    parser.add_argument('--type', default='png', choices=['npy', 'png'], type=str, 
                       help="输入文件类型")
    parser.add_argument('--t', default=None, type=float, 
                       help="背景阈值（仅对npy格式有效）")
    parser.add_argument('--curve', default=False, type=bool, 
                       help="是否生成mIoU曲线")
    parser.add_argument('--num_classes', default=21, type=int, 
                       help="类别数量")
    parser.add_argument('--start', default=35, type=int, 
                       help="阈值搜索起始值（百分比）")
    parser.add_argument('--end', default=55, type=int, 
                       help="阈值搜索结束值（百分比）")
    parser.add_argument('--num_workers', default=64, type=int,
                       help="并行评估的进程数（默认: 64）")
    args = parser.parse_args()

    # 参数验证：npy格式需要指定阈值或启用曲线模式
    if args.type == 'npy':
        assert args.t is not None or args.curve
    
    # 读取图像名称列表
    df = pd.read_csv(args.list, names=['filename'])
    name_list = df['filename'].values
    
    if not args.curve:
        # 单次评估模式：使用指定阈值进行评估
        loglist = do_python_eval(args.predict_dir, args.gt_dir, name_list, args.num_classes, args.type, args.t, printlog=True, num_workers=args.num_workers)
        writelog(args.logfile, loglist, args.comment)
    else:
        # 曲线模式：搜索最佳背景阈值
        l = []           # 存储不同阈值下的mIoU值
        max_mIoU = 0.0   # 最佳mIoU
        best_thr = 0.0   # 最佳阈值
        
        # 在指定范围内搜索最佳阈值
        threshold_range = range(args.start, args.end)
        for i in tqdm(threshold_range, desc="阈值搜索", unit="thr"):
            t = i/100.0  # 将百分比转换为小数
            loglist = do_python_eval(args.predict_dir, args.gt_dir, name_list, args.num_classes, args.type, t, num_workers=args.num_workers)
            l.append(loglist['mIoU'])
            print('%d/%d background score: %.3f\tmIoU: %.3f%%'%(i, args.end-1, t, loglist['mIoU']))
            
            # 更新最佳结果
            if loglist['mIoU'] > max_mIoU:
                max_mIoU = loglist['mIoU']
                best_thr = t
        
        # 输出最佳结果
        print('Best background score: %.3f\tmIoU: %.3f%%' % (best_thr, max_mIoU))
        
        # 将曲线结果写入日志
        writelog(args.logfile, {'mIoU':l, 'Best mIoU': max_mIoU, 'Best threshold': best_thr}, args.comment)

