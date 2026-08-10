import os
import torch
import torch.nn.functional as F
import random
from torchvision import transforms
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
from torch.utils.data import DataLoader
from net.CIDNet import CIDNet
from net.DualSpaceCIDNet import DualSpaceCIDNet  # 双空间CIDNet
from net.DA_MambaNet import DA_MambaNet            # DA-MambaNet（退化感知Mamba）
from data.options import option
from measure import metrics
from eval import eval
from data.data import *
from loss.losses import *
from data.scheduler import *
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter  # TensorBoard支持


opt = option().parse_args()
# opt 是一个 argparse.Namespace 对象
# 这个对象是一个"空的盒子"，可以动态地往里面放东西

def seed_torch(seed = 42):
    """设置所有相关的随机种子以确保实验的可重复性"""
    print(f"使用随机种子：{seed}")
    
    random.seed(seed)           # Python随机数种子
    np.random.seed(seed)        # NumPy随机数种子
    torch.manual_seed(seed)     # PyTorch CPU随机数种子
    torch.cuda.manual_seed(seed)     # PyTorch单GPU随机数种子
    torch.cuda.manual_seed_all(seed) # PyTorch多GPU随机数种子
    os.environ['PYTHONHASHSEED'] = str(seed)  # Python哈希种子，针对dict和set中元素的存储顺序和遍历顺序
    
    # 确保完全可重复性（会稍微降低训练速度）因为显存满了所以更改了记一下
    #torch.backends.cudnn.deterministic = True  # 使用确定性算法
    torch.backends.cudnn.benchmark = True     # 关闭自动优化，确保可重复
    
def train_init():
    """初始化训练环境"""
    seed_torch()                    # 设置随机种子
    # cudnn.benchmark 已在 seed_torch() 中设置为 False 以确保可重复性
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 指定使用第0号GPU
    cuda = opt.gpu_mode
    if cuda and not torch.cuda.is_available():
        raise RuntimeError("No GPU found, please run without --cuda")
    
def train(epoch, writer=None):
    """
    训练一个epoch
    
    Args:
        epoch: 当前epoch编号
        writer: TensorBoard的SummaryWriter对象，用于记录训练过程
        
    
    Returns:
        epoch_loss: 当前epoch的总损失
        batch_count: 处理的batch数量
    """
    model.train()
    epoch_loss = 0      # 累积整个epoch的总损失
    batch_count = 0     # 统计整个epoch处理的batch数量
    train_len = len(training_data_loader)  # DataLoader的长度
    iter = 0            # 当前epoch中已处理的batch计数器
    
    # 梯度累积设置：
    accum_steps = opt.accum_steps  # 累积步数（每accum_steps个batch更新一次参数）
    
    torch.autograd.set_detect_anomaly(opt.grad_detect)
    for batch in tqdm(training_data_loader):
        im1, im2, path1, path2 = batch[0], batch[1], batch[2], batch[3]
        im1 = im1.cuda()
        im2 = im2.cuda()
        
        # use random gamma function (enhancement curve) to improve generalization
        if opt.gamma:
            gamma = random.randint(opt.start_gamma,opt.end_gamma) / 100.0
            input_img = im1 ** gamma
        else:
            input_img = im1
        
        gt_rgb = im2

        # ===================================================================
        # 前向传播（兼容三种模型的不同返回值）
        # CIDNet / DualSpaceCIDNet：直接返回 tensor (B, 3, H, W)
        # DA_MambaNet：返回 (output_rgb, d)，d 是退化条件向量
        # ===================================================================
        if opt.da_mamba:
            # 同一次 DAM 前向同时返回条件概率和原始分类 logits，避免重复计算。
            output_rgb, deg_cond, dam_logits = model(
                input_img, return_dam_logits=True
            )
        else:
            output_rgb = model(input_img)
            deg_cond = None
            dam_logits = None

        output_hvi = model.HVIT(output_rgb)
        gt_hvi = model.HVIT(gt_rgb)
        loss_hvi = L1_loss(output_hvi, gt_hvi) + D_loss(output_hvi, gt_hvi) + E_loss(output_hvi, gt_hvi) + opt.P_weight * P_loss(output_hvi, gt_hvi)[0]
        loss_rgb = L1_loss(output_rgb, gt_rgb) + D_loss(output_rgb, gt_rgb) + E_loss(output_rgb, gt_rgb) + opt.P_weight * P_loss(output_rgb, gt_rgb)[0]
        loss = loss_rgb + opt.HVI_weight * loss_hvi

        # ===================================================================
        # DAM 辅助分类损失（仅在 DA-MambaNet 且 DAM 开启时计算）
        # batch[4] 标签映射：0=低光、1=雾、2=雨、3=雪、4=模糊
        # 若关闭 DAM 或数据集未提供标签，则跳过分类辅助损失。
        # ===================================================================
        if (opt.da_mamba and opt.use_dam and opt.dam_cls_weight > 0
                and dam_logits is not None):
            # batch[4] 为退化类型标签（如有），否则跳过
            if len(batch) > 4 and batch[4] is not None:
                deg_labels = batch[4].cuda().long()   # (B,) 类别标签
                # CrossEntropyLoss 必须接收未经 Softmax 的原始分类分数。
                loss_cls = F.cross_entropy(dam_logits, deg_labels)
                loss = loss + opt.dam_cls_weight * loss_cls
        
        iter += 1
        
        # 梯度累积：损失除以累积步数，保证梯度大小一致
        loss = loss / accum_steps
        loss.backward()  # 累积梯度（不清零）
        
        # 每accum_steps步更新一次参数
        if iter % accum_steps == 0:
            if opt.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01, norm_type=2)
            optimizer.step()
            optimizer.zero_grad()  # 更新后才清零
        
        # 累积损失（还原为原始损失值用于显示）
        epoch_loss += loss.item() * accum_steps
        batch_count += 1
        
        # 每个epoch结束时打印平均损失和学习率，并保存样本图像
        if iter == train_len:
            avg_loss = epoch_loss / batch_count
            current_lr = optimizer.param_groups[0]['lr']
            
            print("===> Epoch[{}]: Loss: {:.4f} || Learning rate: lr={:.6f}".format(
                epoch, avg_loss, current_lr))
            
            # 【TensorBoard记录】记录训练损失和学习率
            if writer is not None:
                writer.add_scalar('Train/Loss', avg_loss, epoch)
                writer.add_scalar('Train/Learning_Rate', current_lr, epoch)
                
                # 记录训练图像（可选）
                # 将第一个batch的第一张图像记录到TensorBoard
                writer.add_image('Train/Output_Image', output_rgb[0], epoch, dataformats='CHW')
                writer.add_image('Train/Ground_Truth', gt_rgb[0], epoch, dataformats='CHW')
            
            # 保存训练样本到本地
            output_img = transforms.ToPILImage()(output_rgb[0].squeeze(0))
            gt_img = transforms.ToPILImage()(gt_rgb[0].squeeze(0))
            if not os.path.exists(opt.val_folder+'training'):          
                os.mkdir(opt.val_folder+'training') 
            output_img.save(opt.val_folder+'training/test.png')
            gt_img.save(opt.val_folder+'training/gt.png')
    
    return epoch_loss, batch_count
                

def checkpoint(epoch):
    """保存模型权重，文件名自动标注超参配置与消融开关"""
    os.makedirs("./weights/train", exist_ok=True)
    if opt.da_mamba:
        # 文件名格式: DAMamba_ds<d_state>_dc<d_conv>_dam<0/1>_film<0/1>_<scan_mode>[_ch24]_epoch_<N>.pth
        tag = f"DAMamba_ds{opt.d_state}_dc{opt.d_conv}_dam{1 if opt.use_dam else 0}_film{1 if opt.use_film else 0}_{opt.scan_mode}"
        if str(opt.channels_mode) == '24':
            tag = f"{tag}_ch24"
        filename = f"{tag}_epoch_{epoch}.pth"
    elif opt.dual_space:
        filename = f"DualSpaceCIDNet_epoch_{epoch}.pth"
    else:
        filename = f"CIDNet_epoch_{epoch}.pth"

    model_out_path = os.path.join("./weights/train", filename)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))
    return model_out_path
    
def load_datasets():
    print('===> Loading datasets')
    if opt.lol_v1 or opt.lol_blur or opt.lolv2_real or opt.lolv2_syn or opt.SID or opt.SICE_mix or opt.SICE_grad or opt.fivek or opt.LoLI_Street or opt.combined_pedestrian:
        if opt.lol_v1:
            train_set = get_lol_training_set(opt.data_train_lol_v1,size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
            test_set = get_eval_set(opt.data_val_lol_v1)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
            
        if opt.lol_blur:
            train_set = get_training_set_blur(opt.data_train_lol_blur,size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
            test_set = get_eval_set(opt.data_val_lol_blur)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)

        if opt.lolv2_real:
            train_set = get_lol_v2_training_set(opt.data_train_lolv2_real,size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
            test_set = get_eval_set(opt.data_val_lolv2_real)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
            
        if opt.lolv2_syn:
            train_set = get_lol_v2_syn_training_set(opt.data_train_lolv2_syn,size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
            test_set = get_eval_set(opt.data_val_lolv2_syn)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
        
        if opt.SID:
            train_set = get_SID_training_set(opt.data_train_SID,size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
            test_set = get_eval_set(opt.data_val_SID)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
            
        if opt.SICE_mix:
            train_set = get_SICE_training_set(opt.data_train_SICE,size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
            test_set = get_SICE_eval_set(opt.data_val_SICE_mix)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
            
        if opt.SICE_grad:
            train_set = get_SICE_training_set(opt.data_train_SICE,size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
            test_set = get_SICE_eval_set(opt.data_val_SICE_grad)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
            
        if opt.fivek:
            train_set = get_fivek_training_set(opt.data_train_fivek,size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
            test_set = get_fivek_eval_set(opt.data_val_fivek)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
        
        if opt.LoLI_Street:
            train_set = get_LoLI_Street_training_set(opt.data_LoLI_Street, size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
            test_set = get_SICE_eval_set(opt.data_val_LoLI_Street)  # 使用 SICE eval（返回4个值）
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
        
        if opt.combined_pedestrian:
            # 天气数据集：Cityscapes雾天 + Cityscapes雨天（已移除 LoLI 低光照）
            train_dirs = [
                opt.data_pedestrian_foggy,  # Cityscapes 雾天行人
                opt.data_pedestrian_rain,   # Cityscapes 雨天行人
            ]
            val_dirs = [
                opt.data_pedestrian_foggy_val,  # Foggy 验证集
                opt.data_pedestrian_rain_val,   # Rain 验证集
            ]
            train_set = get_combined_pedestrian_training_set(train_dirs, size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
            test_set = get_combined_pedestrian_eval_set(val_dirs, eval_size=opt.eval_size)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)

        # （已删除三个单独子数据集的训练代码，统一使用 combined_pedestrian）

    elif opt.allinone:
        # ===== DA-MambaNet 专用：多退化 All-in-One 混合数据集（5类）=====
        print('===> 加载 AllInOne 多退化混合数据集（低光+雾+雨+雪+模糊，共5类）')

        # 解析逗号分隔的路径列表
        lol_dirs  = [d.strip() for d in opt.data_lol_dirs.split(',')  if d.strip()]
        fog_dirs  = [d.strip() for d in opt.data_fog_dirs.split(',')  if d.strip()]
        rain_dirs = [d.strip() for d in opt.data_rain_dirs.split(',') if d.strip()]
        snow_dirs = [d.strip() for d in opt.data_snow_dirs.split(',') if d.strip()]
        blur_dirs = [d.strip() for d in opt.data_blur_dirs.split(',') if d.strip()]

        train_set = get_allinone_training_set(
            lol_dirs  = lol_dirs,
            fog_dirs  = fog_dirs,
            rain_dirs = rain_dirs,
            snow_dirs = snow_dirs,
            blur_dirs = blur_dirs,
            crop_size = opt.cropSize,
            balance   = opt.allinone_balance,
        )
        training_data_loader = DataLoader(
            dataset    = train_set,
            num_workers= opt.threads,
            batch_size = opt.batchSize,
            shuffle    = opt.shuffle,
        )

        # 验证集：5种退化分别验证（通过 opt.max_val_samples 控制截断数量，默认每类最多30张）
        val_dirs   = [opt.data_lol_val, opt.data_fog_val, opt.data_rain_val,
                      opt.data_snow_val, opt.data_blur_val]
        val_labels = [0, 1, 2, 3, 4]   # 0=低光, 1=雾, 2=雨, 3=雪, 4=模糊
        test_set = get_allinone_eval_set(val_dirs, val_labels, max_samples_per_dir=opt.max_val_samples)
        testing_data_loader = DataLoader(
            dataset    = test_set,
            num_workers= opt.threads,
            batch_size = 1,
            shuffle    = False,
        )

    else:
        raise ValueError("should choose a dataset")
    return training_data_loader, testing_data_loader


def build_model():
    """构建模型（支持 CIDNet / DualSpaceCIDNet / DA_MambaNet）"""
    print('===> Building model ')

    # 根据配置选择模型（优先级：da_mamba > dual_space > 原始 CIDNet）
    if opt.da_mamba:
        print('===> 使用 DA-MambaNet（退化感知自适应 Mamba 图像恢复网络）')
        if str(opt.channels_mode) == '24':
            base_channels = [24, 24, 48, 96]
            print('  - 通道模式: 24 (极轻量化 [24,24,48,96], ~2.09M)')
        else:
            base_channels = [36, 36, 72, 144]
            print('  - 通道模式: 36 (标准模式 [36,36,72,144], ~4.45M)')

        print(f'  - num_classes={opt.num_classes}（退化类型数：{opt.num_classes}）')
        print(f'  - d_state={opt.d_state}, d_conv={opt.d_conv}（Mamba SSM 状态维度与因果卷积核）')
        print(f'  - expand=2（固定值）, dam_cls_weight={opt.dam_cls_weight}（固定值）')
        print(f'  - 消融配置: DAM={opt.use_dam}, FiLM={opt.use_film}, scan_mode={opt.scan_mode}')
        if opt.use_rgb_refiner:
            print(f'  - RGB Refiner: 启用（mid_ch={opt.refiner_mid_ch}）')

        model = DA_MambaNet(
            channels       = base_channels,
            num_classes    = opt.num_classes,
            d_state        = opt.d_state,
            d_conv         = opt.d_conv,
            expand         = 2,
            use_rgb_refiner= opt.use_rgb_refiner,
            refiner_mid_ch = opt.refiner_mid_ch,
            use_dam        = opt.use_dam,
            use_film       = opt.use_film,
            scan_mode      = opt.scan_mode,
        ).cuda()
    elif opt.dual_space:
        print('===> 使用 DualSpaceCIDNet (v3: CIDNet + RGB后处理)')
        if opt.use_curve:
            print('===> 启用神经曲线层消融实验 (I通道全局调整)')
        if opt.use_rgb_refiner:
            print(f'===> 启用RGB后处理微调 (mid_ch={opt.refiner_mid_ch})')
        model = DualSpaceCIDNet(
            channels=[36, 36, 72, 144],
            heads=[1, 2, 4, 8],
            use_rgb_refiner=opt.use_rgb_refiner,
            refiner_mid_ch=opt.refiner_mid_ch,
            use_curve=opt.use_curve,
            curve_M=opt.curve_M
        ).cuda()
    else:
        print('===> 使用原始 CIDNet')
        model = CIDNet().cuda()

    # 打印参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f'===> 模型参数量: {total_params/1e6:.3f}M')

    if opt.start_epoch > 0:
        if opt.da_mamba:
            tag = f"DAMamba_ds{opt.d_state}_dc{opt.d_conv}_dam{1 if opt.use_dam else 0}_film{1 if opt.use_film else 0}_{opt.scan_mode}"
            if str(opt.channels_mode) == '24':
                tag = f"{tag}_ch24"
            pth = f"./weights/train/{tag}_epoch_{opt.start_epoch}.pth"
        elif opt.dual_space:
            pth = f"./weights/train/DualSpaceCIDNet_epoch_{opt.start_epoch}.pth"
        else:
            pth = f"./weights/train/CIDNet_epoch_{opt.start_epoch}.pth"

        if not os.path.exists(pth):
            # 回退尝试通用路径
            pth = f"./weights/train/epoch_{opt.start_epoch}.pth"

        if os.path.exists(pth):
            model.load_state_dict(torch.load(pth, map_location=lambda storage, loc: storage))
            print(f'===> 已成功加载预训练模型权重: {pth}')
        else:
            print(f'  [警告] 未找到指定 epoch_{opt.start_epoch} 的预训练权重文件: {pth}')
    return model

def make_scheduler():
    """创建优化器和学习率调度器"""
    # 步骤1: 创建Adam优化器
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)      
    
    # 步骤2: 根据配置选择调度器
    if opt.cos_restart_cyclic:  # 使用循环余弦退火
        # 两段周期：第一段快速探索，第二段精细调整
        phase1 = opt.nEpochs // 4
        remaining = opt.nEpochs - phase1 - opt.start_epoch
        if opt.start_warmup:  # 如果启用warmup
            scheduler_step = CosineAnnealingRestartCyclicLR(
                optimizer=optimizer, 
                periods=[phase1 - opt.warmup_epochs, remaining],
                restart_weights=[1, 1],
                eta_mins=[0.0002, 0.0000001]
            )
            scheduler = GradualWarmupScheduler(
                optimizer, 
                multiplier=1, 
                total_epoch=opt.warmup_epochs, 
                after_scheduler=scheduler_step
            )
        else:
            scheduler = CosineAnnealingRestartCyclicLR(
                optimizer=optimizer, 
                periods=[phase1 - opt.start_epoch, remaining],
                restart_weights=[1, 1],
                eta_mins=[0.0002, 0.0000001])
    elif opt.cos_restart:
        if opt.start_warmup:
            scheduler_step = CosineAnnealingRestartLR(optimizer=optimizer, periods=[opt.nEpochs - opt.warmup_epochs - opt.start_epoch], restart_weights=[1],eta_min=1e-7)
            scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=opt.warmup_epochs, after_scheduler=scheduler_step)
        else:
            scheduler = CosineAnnealingRestartLR(optimizer=optimizer, periods=[opt.nEpochs - opt.start_epoch], restart_weights=[1],eta_min=1e-7)
    else:
        raise ValueError("should choose a scheduler")
    return optimizer,scheduler

def init_loss():
    L1_weight   = opt.L1_weight
    D_weight    = opt.D_weight 
    E_weight    = opt.E_weight 
    P_weight    = 1.0
    
    L1_loss= L1Loss(loss_weight=L1_weight, reduction='mean').cuda() # 创建L1损失函数
    D_loss = SSIM(weight=D_weight).cuda() # 创建SSIM损失函数
    E_loss = EdgeLoss(loss_weight=E_weight).cuda() # 创建边缘损失函数
    P_loss = PerceptualLoss({'conv1_2': 1, 'conv2_2': 1,'conv3_4': 1,'conv4_4': 1}, perceptual_weight = P_weight ,criterion='mse').cuda() # 创建感知损失函数
    return L1_loss,P_loss,E_loss,D_loss 

if __name__ == '__main__':  
    
    '''
    preparision
    '''
    train_init()
    training_data_loader, testing_data_loader = load_datasets()
    model = build_model()
    optimizer,scheduler = make_scheduler()
    L1_loss,P_loss,E_loss,D_loss = init_loss()
    
    '''
    train
    '''
    #峰值信噪比
    psnr = []
    #结构相似性
    ssim = []
    #学习感知图像块相似度（越低越好）
    lpips = []
    start_epoch=0
    if opt.start_epoch > 0:
        start_epoch = opt.start_epoch
    if not os.path.exists(opt.val_folder):          
        os.mkdir(opt.val_folder) #opt.val_folder = 'results/'
    
    # 【TensorBoard初始化】创建TensorBoard写入器
    # 生成带时间戳的日志目录，避免不同实验的日志混在一起
    log_dir = f'./runs/exp{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    writer = SummaryWriter(log_dir)
    print(f"===> TensorBoard日志保存在: {log_dir}")
    print(f"===> 启动TensorBoard: tensorboard --logdir=runs")
    
    
    for epoch in range(start_epoch+1, opt.nEpochs + 1):
        
        # 训练一个epoch，传入writer
        epoch_loss, batch_num = train(epoch, writer=writer)
        scheduler.step()  # 通过调度器更新学习率

        if epoch % opt.snapshots == 0:
            model_out_path = checkpoint(epoch) #每隔opt.snapshots个epoch保存一次模型
            # 在验证集上评估模型性能
            norm_size = True #是否将图像归一化（统一）到固定尺寸

            # LOL three subsets
            if opt.lol_v1:
                output_folder = 'LOLv1/'#模型生成的增强图像保存路径，保存在results/LOLv1/文件夹下
                label_dir = opt.data_valgt_lol_v1#验证集真实图像保存路径，保存在datasets/LOLdataset/eval15/high/文件夹下
            if opt.lolv2_real:
                output_folder = 'LOLv2_real/'
                label_dir = opt.data_valgt_lolv2_real
            if opt.lolv2_syn:
                output_folder = 'LOLv2_syn/'
                label_dir = opt.data_valgt_lolv2_syn
                
            # LOL-blur dataset with low_blur and high_sharp_scaled
            if opt.lol_blur:
                output_folder = 'LOL_blur/'
                label_dir = opt.data_valgt_lol_blur
                    
            if opt.SID:
                output_folder = 'SID/'
                label_dir = opt.data_valgt_SID
                npy = True #没用到
            if opt.SICE_mix:
                output_folder = 'SICE_mix/'
                label_dir = opt.data_valgt_SICE_mix
                norm_size = False
            if opt.SICE_grad:
                output_folder = 'SICE_grad/'
                label_dir = opt.data_valgt_SICE_grad
                norm_size = False
                    
            if opt.fivek:
                output_folder = 'fivek/'
                label_dir = opt.data_valgt_fivek
                norm_size = False
            if opt.LoLI_Street:
                output_folder = 'LoLI_Street/'
                label_dir = opt.data_valgt_LoLI_Street
                norm_size = False
            if opt.combined_pedestrian:
                output_folder = 'combined_pedestrian/'
                norm_size = False
            if opt.allinone:
                output_folder = 'allinone/'
                norm_size = False
            
            im_dir = opt.val_folder + output_folder  # 只传目录路径
            
            # 清空上一次 eval 的输出图片，防止旧文件（如 LoLI）残留影响指标计算
            if os.path.exists(im_dir):
                import shutil
                for f in os.listdir(im_dir):
                    fp = os.path.join(im_dir, f)
                    if os.path.isfile(fp):
                        os.remove(fp)
                print(f"  ✅ 已清空旧输出: {im_dir}")
            
            # 每隔一定epoch进行模型评估
            eval(model, testing_data_loader, model_out_path, opt.val_folder+output_folder, 
                    norm_size=norm_size, LOL=opt.lol_v1, v2=opt.lolv2_real, alpha=0.8)
            
            # ===== 计算评估指标 =====
            if opt.allinone:
                # AllInOne 多退化混合验证集：自动汇总 5 个退化验证集的 GT 到临时目录计算总指标
                import shutil
                val_gt_dirs = [
                    opt.data_lol_val,
                    opt.data_fog_val,
                    opt.data_rain_val,
                    opt.data_snow_val,
                    opt.data_blur_val
                ]
                combined_gt_dir = os.path.join(im_dir, '_temp_combined_gt')
                os.makedirs(combined_gt_dir, exist_ok=True)
                for v_dir in val_gt_dirs:
                    if not os.path.exists(v_dir):
                        continue
                    high_dir = os.path.join(v_dir, 'high')
                    prefix = os.path.basename(v_dir)
                    if os.path.isdir(high_dir):
                        files = sorted([f for f in os.listdir(high_dir) if is_image_file(f)])
                        if opt.max_val_samples and opt.max_val_samples > 0:
                            files = files[:opt.max_val_samples]
                        for f in files:
                            src = os.path.join(high_dir, f)
                            dst_name = f"{prefix}_{f}"
                            shutil.copy2(src, os.path.join(combined_gt_dir, dst_name))
                
                print("\n--- AllInOne 5类多退化混合验证指标 ---")
                avg_psnr, avg_ssim, avg_lpips = metrics(
                    im_dir, combined_gt_dir + '/', use_GT_mean=False
                )
                shutil.rmtree(combined_gt_dir, ignore_errors=True)
            elif opt.combined_pedestrian:
                # 合并数据集验证：直接在整体合并的 GT 上计算总指标（不再拆分子集）
                import shutil
                
                # 两个天气子数据集的 GT 目录（已移除 loli）
                gt_dirs = {
                    'foggy': opt.data_pedestrian_foggy_val + '/high/',
                    'rain': opt.data_pedestrian_rain_val + '/high/',
                }
                
                # 构建合并 GT 临时目录：将三个子集的 GT 图片汇聚到一个目录
                combined_gt_dir = os.path.join(im_dir, '_temp_combined_gt')
                os.makedirs(combined_gt_dir, exist_ok=True)
                for key, gt_dir in gt_dirs.items():
                    for f in os.listdir(gt_dir):
                        src = os.path.join(gt_dir, f)
                        if os.path.isfile(src):
                            shutil.copy2(src, os.path.join(combined_gt_dir, f))
                
                print("\n--- 整体 (Combined) 验证指标 ---")
                avg_psnr, avg_ssim, avg_lpips = metrics(
                    im_dir, combined_gt_dir + '/', use_GT_mean=False
                )
                
                # 清理合并GT临时目录
                shutil.rmtree(combined_gt_dir, ignore_errors=True)
            else:
                # 非合并数据集：直接计算
                avg_psnr, avg_ssim, avg_lpips = metrics(im_dir, label_dir, use_GT_mean=False)
            
            print("===> Avg.PSNR: {:.4f} dB ".format(avg_psnr))
            print("===> Avg.SSIM: {:.4f} ".format(avg_ssim))
            print("===> Avg.LPIPS: {:.4f} ".format(avg_lpips))
                
            # 保存指标到列表（使用整体指标作为模型选择依据）
            psnr.append(avg_psnr)
            ssim.append(avg_ssim)
            lpips.append(avg_lpips)
                
            # 【TensorBoard记录】记录整体评估指标
            writer.add_scalar('Eval/PSNR', avg_psnr, epoch)
            writer.add_scalar('Eval/SSIM', avg_ssim, epoch)
            writer.add_scalar('Eval/LPIPS', avg_lpips, epoch)
                
            # 同时在一个图中显示所有指标的变化趋势
            writer.add_scalars('Eval/All_Metrics', {
                'PSNR': avg_psnr,
                'SSIM': avg_ssim * 30,
                'LPIPS': avg_lpips * 100,
            }, epoch)
            print(psnr)
            print(ssim)
            print(lpips)
            
        
        torch.cuda.empty_cache()
    # 【训练完成】关闭TensorBoard写入器
    print("\n===> 训练完成！")
    
    # 记录最终的最佳结果到TensorBoard
    if len(psnr) > 0:
        best_psnr = max(psnr)
        best_ssim = max(ssim)
        best_lpips = min(lpips)
        best_psnr_epoch = (psnr.index(best_psnr) + 1) * opt.snapshots
        best_ssim_epoch = (ssim.index(best_ssim) + 1) * opt.snapshots
        best_lpips_epoch = (lpips.index(best_lpips) + 1) * opt.snapshots
        
        writer.add_text('Final_Results/Best_PSNR', f'{best_psnr:.4f} at Epoch {best_psnr_epoch}')
        writer.add_text('Final_Results/Best_SSIM', f'{best_ssim:.4f} at Epoch {best_ssim_epoch}')
        writer.add_text('Final_Results/Best_LPIPS', f'{best_lpips:.4f} at Epoch {best_lpips_epoch}')
    
    writer.close()
    print(f"===> TensorBoard日志已保存到: {log_dir}")
    
    # 保存所有指标到Markdown文件
    now = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    os.makedirs("./results/metrics", exist_ok=True)
    with open(f"./results/metrics/metrics{now}.md", "w", encoding="utf-8") as f:
        f.write("# DA-MambaNet 训练评估报告 (Metrics Report)\n\n")
        f.write(f"- **评估时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **输出目录**: `{output_folder}`\n")
        f.write(f"- **TensorBoard 日志**: `{log_dir}`\n\n")
        
        f.write("## 1. 结构消融实验开关 (Ablation Switches)\n\n")
        f.write("| 消融维度 | 配置变量 | 当前值 | 含义/设计说明 |\n")
        f.write("|---------|---------|-------|--------------|\n")
        f.write(f"| **退化感知** | `use_dam` | `{opt.use_dam}` | 是否开启 DAM 的5类概率与连续退化潜变量条件 |\n")
        f.write(f"| **条件调制** | `use_film` | `{opt.use_film}` | 是否开启 FiLM 特征仿射变换调制 |\n")
        f.write(f"| **扫描策略** | `scan_mode` | `{opt.scan_mode}` | `hetero`(HV-2向/I-4向), `all_2way`, `all_4way` |\n")
        f.write(f"| **RGB后处理** | `use_rgb_refiner` | `{opt.use_rgb_refiner}` | RGB 空间残差微调（常驻基础模块） |\n\n")

        f.write("## 2. 超参数敏感性分析配置 (Hyperparameter Sensitivity)\n\n")
        f.write("| 超参数 | 配置变量 | 当前值 | 说明与取值范围 |\n")
        f.write("|-------|---------|-------|--------------|\n")
        f.write(f"| **SSM 隐状态维度** | `d_state` | `{opt.d_state}` | 长距离记忆容量 N (可选: 8, 16, 32) |\n")
        f.write(f"| **因果卷积核大小** | `d_conv` | `{opt.d_conv}` | 短程局部感受野 K (可选: 2, 3, 4, 5) |\n\n")

        f.write("## 2.1 固定超参数 (Fixed Hyperparameters)\n\n")
        f.write("| 超参数 | 配置变量 | 固定值 | 固定原因 |\n")
        f.write("|-------|---------|-------|---------|\n")
        f.write(f"| **通道扩展倍数** | `expand` | `2` | 与 channels_mode 冗余，Mamba 系列论文均固定为 2 |\n")
        f.write(f"| **分类辅助权重** | `dam_cls_weight` | `{opt.dam_cls_weight}` | 训练策略参数，AllInOneDataset 自带标签，固定 0.1 |\n")
        f.write(f"| **通道规模模式** | `channels_mode` | `{opt.channels_mode}` | 架构设计常量 (36: [36,36,72,144] ~4.45M) |\n\n")

        f.write("## 3. 基础训练超参数 (Training Hyperparameters)\n\n")
        f.write(f"- **学习率 (lr)**: `{opt.lr}`\n")
        f.write(f"- **批次大小 (batchSize)**: `{opt.batchSize}`\n")
        f.write(f"- **梯度累加 (accum_steps)**: `{opt.accum_steps}`\n")
        f.write(f"- **裁剪尺寸 (cropSize)**: `{opt.cropSize}`\n")
        f.write(f"- **验证集每类上限 (max_val_samples)**: `{opt.max_val_samples}`\n")
        f.write(f"- **损失函数权重**: HVI=`{opt.HVI_weight}`, L1=`{opt.L1_weight}`, D=`{opt.D_weight}`, E=`{opt.E_weight}`, P=`{opt.P_weight}`\n\n")
        
        # 最佳结果汇总
        best_psnr_idx = psnr.index(max(psnr))
        best_ssim_idx = ssim.index(max(ssim))
        best_lpips_idx = lpips.index(min(lpips))
        
        f.write("## 最佳结果\n\n")
        f.write(f"- **最佳PSNR**: {max(psnr):.4f} (Epoch {(best_psnr_idx+1)*opt.snapshots})\n")
        f.write(f"- **最佳SSIM**: {max(ssim):.4f} (Epoch {(best_ssim_idx+1)*opt.snapshots})\n")
        f.write(f"- **最低LPIPS**: {min(lpips):.4f} (Epoch {(best_lpips_idx+1)*opt.snapshots})\n\n")
        
        # 整体指标表格
        f.write("## 整体 (Combined) 指标\n\n")
        f.write("| Epochs | PSNR | SSIM | LPIPS |\n")
        f.write("|--------|------|------|-------|\n")
        for i in range(len(psnr)):
            f.write(f"| {opt.start_epoch+(i+1)*opt.snapshots} | {psnr[i]:.4f} | {ssim[i]:.4f} | {lpips[i]:.4f} |\n")
        
        # 分子集指标表格已移除（验证阶段仅计算整体指标）
        
        f.write(f"\n## 最终结果（整体）\n\n")
        f.write(f"| 指标 | 最佳值 | 对应Epoch |\n")
        f.write(f"|------|--------|----------|\n")
        f.write(f"| PSNR ↑ | {max(psnr):.4f} | {(best_psnr_idx+1)*opt.snapshots} |\n")
        f.write(f"| SSIM ↑ | {max(ssim):.4f} | {(best_ssim_idx+1)*opt.snapshots} |\n")
        f.write(f"| LPIPS ↓ | {min(lpips):.4f} | {(best_lpips_idx+1)*opt.snapshots} |\n")
