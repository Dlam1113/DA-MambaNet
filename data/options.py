import argparse

def option():
    # Training settings
    # 创建ArgumentParser对象
    parser = argparse.ArgumentParser(description='CIDNet')

    # 添加各种命令行参数
    parser.add_argument('--batchSize', type=int, default=8, help='training batch size')
    parser.add_argument('--cropSize', type=int, default=256, help='image crop size (patch size)')
    parser.add_argument('--eval_size', type=int, default=512, help='image resize size for evaluation (0 for original size)')
    parser.add_argument('--nEpochs', type=int, default=500, help='number of epochs to train for end')
    parser.add_argument('--start_epoch', type=int, default=0, help='number of epochs to start, >0 is retrained a pre-trained pth')
    parser.add_argument('--snapshots', type=int, default=1, help='Snapshots for save checkpoints pth')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate')
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--threads', type=int, default=16, help='number of threads for dataloader to use')
    parser.add_argument('--accum_steps', type=int, default=1, help='gradient accumulation steps')

    # choose a scheduler 学习率调度器的作用是使学习率周期性变化帮助模型跳过局部最优解
    parser.add_argument('--cos_restart_cyclic', type=bool, default=True)
    parser.add_argument('--cos_restart', type=bool, default=False)

    # warmup training
    parser.add_argument('--warmup_epochs', type=int, default=3, help='warmup_epochs')
    parser.add_argument('--start_warmup', type=bool, default=True, help='turn False to train without warmup') 

    # train datasets  训练数据路径
    parser.add_argument('--data_train_lol_blur'     , type=str, default='./datasets/LOL_blur/train')
    parser.add_argument('--data_train_lol_v1'       , type=str, default='./datasets/LOLdataset/our485')
    parser.add_argument('--data_train_lolv2_real'   , type=str, default='./datasets/LOLv2/Real_captured/Train')
    parser.add_argument('--data_train_lolv2_syn'    , type=str, default='./datasets/LOLv2/Synthetic/Train')
    parser.add_argument('--data_train_SID'          , type=str, default='./datasets/Sony_total_dark/train')
    parser.add_argument('--data_train_SICE'         , type=str, default='./datasets/SICE/Dataset/train')
    parser.add_argument('--data_train_fivek'        , type=str, default='./datasets/FiveK/train')
    parser.add_argument('--data_LoLI_Street'        , type=str, default='./filtered/loli_pedestrian',
                        help='LoLI-Street训练数据根目录（下含 low/ 和 high/）') 

    # ========== 合并行人数据集路径（服务器上 filtered/ 在 HVI-CIDNet-master 下） ==========
    parser.add_argument('--data_pedestrian_loli'    , type=str, default='./filtered/loli_pedestrian')
    parser.add_argument('--data_pedestrian_foggy'   , type=str, default='./filtered/cityscapes_foggy_pedestrian')
    parser.add_argument('--data_pedestrian_rain'    , type=str, default='./filtered/cityscapes_rain_pedestrian')
    parser.add_argument('--data_pedestrian_loli_val' , type=str, default='./filtered/loli_pedestrian_val')
    parser.add_argument('--data_pedestrian_foggy_val', type=str, default='./filtered/cityscapes_foggy_pedestrian_val')
    parser.add_argument('--data_pedestrian_rain_val' , type=str, default='./filtered/cityscapes_rain_pedestrian_val')

    # validation input   验证输入路径
    parser.add_argument('--data_val_lol_blur'       , type=str, default='./datasets/LOL_blur/eval/low_blur')
    parser.add_argument('--data_val_lol_v1'         , type=str, default='./datasets/LOLdataset/eval15/low')
    parser.add_argument('--data_val_lolv2_real'     , type=str, default='./datasets/LOLv2/Real_captured/Test/Low')
    parser.add_argument('--data_val_lolv2_syn'      , type=str, default='./datasets/LOLv2/Synthetic/Test/Low')
    parser.add_argument('--data_val_SID'            , type=str, default='./datasets/Sony_total_dark/eval/short')
    parser.add_argument('--data_val_SICE_mix'       , type=str, default='./datasets/SICE/Dataset/eval/test')
    parser.add_argument('--data_val_SICE_grad'      , type=str, default='./datasets/SICE/Dataset/eval/test')
    parser.add_argument('--data_test_fivek'         , type=str, default='./datasets/FiveK/test/input')
    parser.add_argument('--data_val_LoLI_Street'    , type=str, default='./filtered/loli_pedestrian_val/low',
                        help='LoLI-Street验证集输入（low目录）')

    # validation groundtruth   验证真值路径
    parser.add_argument('--data_valgt_lol_blur'     , type=str, default='./datasets/LOL_blur/eval/high_sharp_scaled/')
    parser.add_argument('--data_valgt_lol_v1'       , type=str, default='./datasets/LOLdataset/eval15/high/')
    parser.add_argument('--data_valgt_lolv2_real'   , type=str, default='./datasets/LOLv2/Real_captured/Test/Normal/')
    parser.add_argument('--data_valgt_lolv2_syn'    , type=str, default='./datasets/LOLv2/Synthetic/Test/Normal/')
    parser.add_argument('--data_valgt_SID'          , type=str, default='./datasets/Sony_total_dark/eval/long/')
    parser.add_argument('--data_valgt_SICE_mix'     , type=str, default='./datasets/SICE/Dataset/eval/target/')
    parser.add_argument('--data_valgt_SICE_grad'    , type=str, default='./datasets/SICE/Dataset/eval/target/')
    parser.add_argument('--data_valgt_fivek'        , type=str, default='./datasets/FiveK/test/target/')
    parser.add_argument('--data_valgt_LoLI_Street'  , type=str, default='./filtered/loli_pedestrian_val/high',
                        help='LoLI-Street验证集GT（high目录）')

    parser.add_argument('--val_folder', default='./results/', help='Location to save validation datasets')

    # loss weights 损失权重
    parser.add_argument('--HVI_weight', type=float, default=1.0)
    parser.add_argument('--L1_weight', type=float, default=1.0)
    parser.add_argument('--D_weight',  type=float, default=0.5)
    parser.add_argument('--E_weight',  type=float, default=50.0)
    parser.add_argument('--P_weight',  type=float, default=1e-2)
    
    # use random gamma function (enhancement curve) to improve generalization 使用随机gamma函数提高泛化能力
    parser.add_argument('--gamma', type=bool, default=True)
    parser.add_argument('--start_gamma', type=int, default=60)
    parser.add_argument('--end_gamma', type=int, default=120)

    # auto grad, turn off to speed up training
    parser.add_argument('--grad_detect', type=bool, default=True)  # 梯度爆炸检测
    parser.add_argument('--grad_clip', type=bool, default=True)     # 梯度裁剪
    
    # ========== 双空间CIDNet配置 ==========
    parser.add_argument('--dual_space', type=bool, default=False, 
                        help='是否使用DualSpaceCIDNet（v3: CIDNet + RGB后处理）')
    
    # ========== RGB后处理配置 ==========
    parser.add_argument('--use_rgb_refiner', type=bool, default=False,
                        help='是否启用RGB后处理微调（消融实验可关闭）')
    parser.add_argument('--refiner_mid_ch', type=int, default=64,
                        help='RGB Refiner中间层通道数')
    
    # ========== 神经曲线层消融实验 ==========
    parser.add_argument('--use_curve', type=bool, default=False,
                        help='是否使用神经曲线层对I通道进行全局调整（消融实验）')
    parser.add_argument('--curve_M', type=int, default=11,
                        help='曲线控制点数量')
    
    
    # choose which dataset you want to train, please only set one "True"
    parser.add_argument('--lol_v1', type=bool, default=False)
    parser.add_argument('--lolv2_real', type=bool, default=False)
    parser.add_argument('--lolv2_syn', type=bool, default=False)
    parser.add_argument('--lol_blur', type=bool, default=False)
    parser.add_argument('--SID', type=bool, default=False)
    parser.add_argument('--SICE_mix', type=bool, default=False)
    parser.add_argument('--SICE_grad', type=bool, default=False)
    parser.add_argument('--fivek', type=bool, default=False)
    parser.add_argument('--LoLI_Street', type=bool, default=False,
                        help='仅使用 LoLI-Street 低光照数据集训练与验证')
    parser.add_argument('--combined_pedestrian', type=bool, default=False,
                        help='使用合并行人数据集（LoLI低光照+Cityscapes雾天+雨天）')

    # ========== DA-MambaNet 专用配置 ==========
    # DA-MambaNet 是本项目的核心模型，使用退化感知 + Mamba 状态空间模型的混合架构
    # 设为 True 时会替代 CIDNet / DualSpaceCIDNet 进行训练
    parser.add_argument('--da_mamba', type=bool, default=True,
                        help='是否使用 DA-MambaNet（退化感知自适应 Mamba 图像恢复网络）')
    # 退化类型数量，对应 DAM 模块的分类头输出维度
    # 必须与 allinone_dataset.py 中的标签映射一致：0=低光, 1=雾, 2=雨, 3=雪, 4=模糊
    parser.add_argument('--num_classes', type=int, default=5,
                        help='退化类型分类数（5类: 低光/雾/雨/雪/模糊）')
    # Mamba SSM 的隐状态维度，控制模型对长距离依赖的记忆容量
    # 推荐值：16（轻量）或 32（更强表达力但显存占用增大）
    parser.add_argument('--d_state', type=int, default=16,
                        help='Mamba SSM 状态空间维度（越大记忆容量越强，计算量越大）')
    # DAM 分类辅助损失：利用退化类型标签监督 DAM 模块的退化分类
    # 设为 0 时 DAM 仅通过主恢复损失间接学习退化表征（无监督模式）
    # 设为 0.1 时加入交叉熵辅助损失，加速 DAM 收敛（需要数据集提供标签）
    parser.add_argument('--dam_cls_weight', type=float, default=0.0,
                        help='DAM 分类辅助损失权重（0表示禁用，推荐有标签时设为0.1）')

    # ========== DA-MambaNet 一键消融实验配置 ==========
    parser.add_argument('--use_dam', type=bool, default=True,
                        help='【消融】是否启用 DAM 退化感知模块（设为 False 时使用零向量代替条件向量）')
    parser.add_argument('--use_film', type=bool, default=True,
                        help='【消融】是否启用 FiLM 特征条件调制（设为 False 时旁路特征调制）')
    parser.add_argument('--scan_mode', type=str, default='hetero',
                        help='【消融】Mamba 扫描模式：hetero(HV流2向/I流4向异构，默认), all_2way(全2向), all_4way(全4向)')
    parser.add_argument('--channels_mode', type=str, default='36',
                        help='【消融/轻量化】通道数模式：36(即[36,36,72,144], 4.45M), 24(即[24,24,48,96], 2.09M极轻量)')

    # ========== AllInOne 混合数据集路径（DA-MambaNet 专用）==========
    # AllInOne 模式：将 5 种退化类型的数据集混合为一个统一的训练集
    # 每个样本会携带退化类型标签（0-4），供 DAM 分类辅助损失使用
    # 与 --da_mamba True 配合使用，实现"一个模型处理所有退化"的 All-in-One 训练
    parser.add_argument('--allinone', type=bool, default=True,
                        help='是否使用 AllInOne 混合数据集（低光+雾+雨+雪+模糊五合一）')
    # 平衡模式：对各退化类别进行过采样/欠采样，使每种类型的样本数相同
    # 适用于各退化数据集样本数差异较大的情况（如 LOLv1=485 vs GoPro=2103）
    parser.add_argument('--allinone_balance', type=bool, default=False,
                        help='是否按退化类别平衡 AllInOne 数据集')
    # 低光照训练集（支持多个路径，用逗号分隔）
    parser.add_argument('--data_lol_dirs', type=str,
                        default='./datasets/LOLv1/train',
                        help='低光照训练集路径，多个用逗号分隔')
    # 雾天训练集
    parser.add_argument('--data_fog_dirs', type=str,
                        default='./datasets/Fog_train',
                        help='雾天训练集路径，多个用逗号分隔')
    # 雨天训练集
    parser.add_argument('--data_rain_dirs', type=str,
                        default='./datasets/Rain_train',
                        help='雨天训练集路径，多个用逗号分隔')
    # 雪天训练集
    parser.add_argument('--data_snow_dirs', type=str,
                        default='./datasets/Snow_train',
                        help='雪天训练集路径，多个用逗号分隔')
    # 运动模糊训练集
    parser.add_argument('--data_blur_dirs', type=str,
                        default='./datasets/GoPro_train',
                        help='运动模糊训练集路径，多个用逗号分隔')
    # 验证集（每种退化类型一个路径）
    parser.add_argument('--data_lol_val', type=str,
                        default='./datasets/LOLv1/eval15',
                        help='低光照验证集路径')
    parser.add_argument('--data_fog_val', type=str,
                        default='./datasets/Fog_val',
                        help='雾天验证集路径')
    parser.add_argument('--data_rain_val', type=str,
                        default='./datasets/Rain_val',
                        help='雨天验证集路径')
    parser.add_argument('--data_snow_val', type=str,
                        default='./datasets/Snow_val',
                        help='雪天验证集路径')
    parser.add_argument('--data_blur_val', type=str,
                        default='./datasets/GoPro_val',
                        help='运动模糊验证集路径')
    parser.add_argument('--max_val_samples', type=int, default=30,
                        help='验证集每种退化类型的最大评估样本数（默认30，LOLv1全用15张，其余4类截断前30张；0或-1表示不限制）')

    return parser
