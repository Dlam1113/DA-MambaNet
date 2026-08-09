import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
from tqdm import tqdm
from data.data import *
from torchvision import transforms
from torch.utils.data import DataLoader
from loss.losses import *
from net.CIDNet import CIDNet

eval_parser = argparse.ArgumentParser(description='Eval')
eval_parser.add_argument('--perc', action='store_true', help='trained with perceptual loss')
eval_parser.add_argument('--lol', action='store_true', help='output lolv1 dataset')
eval_parser.add_argument('--lol_v2_real', action='store_true', help='output lol_v2_real dataset')
eval_parser.add_argument('--lol_v2_syn', action='store_true', help='output lol_v2_syn dataset')
eval_parser.add_argument('--SICE_grad', action='store_true', help='output SICE_grad dataset')
eval_parser.add_argument('--SICE_mix', action='store_true', help='output SICE_mix dataset')
eval_parser.add_argument('--fivek', action='store_true', help='output FiveK dataset')

eval_parser.add_argument('--best_GT_mean', action='store_true', help='output lol_v2_real dataset best_GT_mean')
eval_parser.add_argument('--best_PSNR', action='store_true', help='output lol_v2_real dataset best_PSNR')
eval_parser.add_argument('--best_SSIM', action='store_true', help='output lol_v2_real dataset best_SSIM')

eval_parser.add_argument('--custome', action='store_true', help='output custome dataset')
eval_parser.add_argument('--custome_path', type=str, default='./YOLO')
eval_parser.add_argument('--unpaired', action='store_true', help='output unpaired dataset')
eval_parser.add_argument('--DICM', action='store_true', help='output DICM dataset')
eval_parser.add_argument('--LIME', action='store_true', help='output LIME dataset')
eval_parser.add_argument('--MEF', action='store_true', help='output MEF dataset')
eval_parser.add_argument('--NPE', action='store_true', help='output NPE dataset')
eval_parser.add_argument('--VV', action='store_true', help='output VV dataset')
eval_parser.add_argument('--alpha', type=float, default=1.0)
eval_parser.add_argument('--gamma', type=float, default=1.0)
eval_parser.add_argument('--unpaired_weights', type=str, default='./weights/LOLv2_syn/w_perc.pth')

ep, _ = eval_parser.parse_known_args()


def eval(model, testing_data_loader, model_path, output_folder, norm_size=True, LOL=False, v2=False, unpaired=False,
         alpha=1.0, gamma=1.0):
    """
    模型评估与推理图片保存函数 (Evaluation & Inference Save Function)
    
    对测试/验证数据集中的图片进行模型前向推理，并将模型生成的图像结果保存到指定文件夹中。
    
    参数说明 (Parameters):
        model (nn.Module): 待评估的深度学习模型实例（如 CIDNet、DualSpaceCIDNet、DA_MambaNet）。
        testing_data_loader (DataLoader): 验证集的数据加载器，按 Batch 视角读取测试图像。
        model_path (str): 模型保存的权重文件路径（.pth 文件）。
        output_folder (str): 评估生成的预测图片保存的目标文件夹路径。
        norm_size (bool): 图像尺寸是否已经归一化/裁剪好。若为 False，则需要按原始 h, w 裁切恢复尺寸。
        LOL (bool): 是否为 LOLv1 数据集特定推断开关。
        v2 (bool): 是否为 LOLv2 数据集特定推断开关。
        unpaired (bool): 是否为未配对数据集推理模式。
        alpha (float): 调节系数（用于模型内部某些增强通道）。
        gamma (float): 增强曲线 Gamma 值的指数次方（默认为 1.0 表示不调整）。
    """
    torch.set_grad_enabled(False)
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    print('Pre-trained model is loaded.')
    model.eval()
    print('Evaluation:')
    if LOL:
        model.trans.gated = True
    elif v2:
        model.trans.gated2 = True
        model.trans.alpha = alpha
    elif unpaired:
        model.trans.gated2 = True
        model.trans.alpha = alpha
        
    for batch in tqdm(testing_data_loader):
        with torch.no_grad():
            if norm_size:
                input, name = batch[0], batch[1]
            else:
                input, name, h, w = batch[0], batch[1], batch[2], batch[3]
            
            input = input.cuda()
            result = model(input**gamma)
            
            # =====================================================================
            # 解包模型输出 (兼容多种网络架构的返回值):
            # 1. dict 结构 (例如 DualSpaceCIDNet 返回 {'output': Tensor, ...}): 提取 ['output']
            # 2. tuple/list 结构 (例如 DA_MambaNet 返回 (output_rgb, deg_cond)): 提取第 0 项 output_rgb
            # 3. 单张量 Tensor 结构 (例如 标准 CIDNet 返回 output_rgb): 直接使用
            # =====================================================================
            if isinstance(result, dict):
                output = result['output']
            elif isinstance(result, (tuple, list)):
                output = result[0]
            else:
                output = result
            
        if not os.path.exists(output_folder):          
            os.mkdir(output_folder)  
            
        output = torch.clamp(output, 0, 1).cuda()  # 将output中的像素数值截断在 [0, 1] 合法RGB范围之间
        if not norm_size:
            output = output[:, :, :h, :w] # 恢复到原来图片尺寸
        
        output_img = transforms.ToPILImage()(output.squeeze(0))
        output_img.save(output_folder + name[0]) # 因为 name 是元组类型所以必须取索引 [0]
        torch.cuda.empty_cache()  # 清理 GPU 显存缓存
    print('===> End evaluation')
    if LOL:
        model.trans.gated = False
    elif v2:
        model.trans.gated2 = False
    torch.set_grad_enabled(True)
    
if __name__ == '__main__':
    
    cuda = True
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, or need to change CUDA_VISIBLE_DEVICES number")
    
    if not os.path.exists('./output'):          
            os.mkdir('./output')  
    
    norm_size = True
    num_workers = 1
    alpha = None
    if ep.lol:
        eval_data = DataLoader(dataset=get_eval_set("./datasets/LOLdataset/eval15/low"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './output/LOLv1/'
        if ep.perc:
            weight_path = './weights/LOLv1/w_perc.pth'
        else:
            weight_path = './weights/LOLv1/wo_perc.pth'
        
            
    elif ep.lol_v2_real:
        eval_data = DataLoader(dataset=get_eval_set("./datasets/LOLv2/Real_captured/Test/Low"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './output/LOLv2_real/'
        if ep.best_GT_mean:
            weight_path = './weights/LOLv2_real/w_perc.pth'
            alpha = 0.84
        elif ep.best_PSNR:
            weight_path = './weights/LOLv2_real/best_PSNR.pth'
            alpha = 0.8
        elif ep.best_SSIM:
            weight_path = './weights/LOLv2_real/best_SSIM.pth'
            alpha = 0.82
            
    elif ep.lol_v2_syn:
        eval_data = DataLoader(dataset=get_eval_set("./datasets/LOLv2/Synthetic/Test/Low"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './output/LOLv2_syn/'
        if ep.perc:
            weight_path = './weights/LOLv2_syn/w_perc.pth'
        else:
            weight_path = './weights/LOLv2_syn/wo_perc.pth'
            
    elif ep.SICE_grad:
        eval_data = DataLoader(dataset=get_SICE_eval_set("./datasets/SICE/SICE_Grad"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './output/SICE_grad/'
        weight_path = './weights/SICE.pth'
        norm_size = False
        
    elif ep.SICE_mix:
        eval_data = DataLoader(dataset=get_SICE_eval_set("./datasets/SICE/SICE_Mix"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './output/SICE_mix/'
        weight_path = './weights/SICE.pth'
        norm_size = False
        
    elif ep.fivek:
        eval_data = DataLoader(dataset=get_SICE_eval_set("./datasets/FiveK/test/input"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './output/fivek/'
        weight_path = './weights/fivek.pth'
        norm_size = False
    
    elif ep.unpaired: 
        if ep.DICM:
            eval_data = DataLoader(dataset=get_SICE_eval_set("./datasets/DICM"), num_workers=num_workers, batch_size=1, shuffle=False)
            output_folder = './output/DICM/'
        elif ep.LIME:
            eval_data = DataLoader(dataset=get_SICE_eval_set("./datasets/LIME"), num_workers=num_workers, batch_size=1, shuffle=False)
            output_folder = './output/LIME/'
        elif ep.MEF:
            eval_data = DataLoader(dataset=get_SICE_eval_set("./datasets/MEF"), num_workers=num_workers, batch_size=1, shuffle=False)
            output_folder = './output/MEF/'
        elif ep.NPE:
            eval_data = DataLoader(dataset=get_SICE_eval_set("./datasets/NPE"), num_workers=num_workers, batch_size=1, shuffle=False)
            output_folder = './output/NPE/'
        elif ep.VV:
            eval_data = DataLoader(dataset=get_SICE_eval_set("./datasets/VV"), num_workers=num_workers, batch_size=1, shuffle=False)
            output_folder = './output/VV/'
        elif ep.custome:
            eval_data = DataLoader(dataset=get_SICE_eval_set(ep.custome_path), num_workers=num_workers, batch_size=1, shuffle=False)
            output_folder = './output/custome/'
        alpha = ep.alpha
        norm_size = False
        weight_path = ep.unpaired_weights
        
    eval_net = CIDNet().cuda()
    eval(eval_net, eval_data, weight_path, output_folder,norm_size=norm_size,LOL=ep.lol,v2=ep.lol_v2_real,unpaired=ep.unpaired,alpha=alpha,gamma=ep.gamma)

