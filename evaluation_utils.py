"""训练期快速验证的纯 Python 汇总与调度工具。"""


QUICK_VAL_CLASS_NAMES = ('lowlight', 'fog', 'rain', 'snow', 'blur')


def should_run_evaluation(epoch, total_epochs, snapshots):
    """判断当前 epoch 是否应执行周期评估或训练结束兜底评估。"""
    if snapshots <= 0:
        raise ValueError(f"snapshots 必须大于0，当前值为: {snapshots}")
    return epoch % snapshots == 0 or epoch == total_epochs


def resolve_quick_val_class(filename, class_prefixes):
    """根据输出文件名前缀解析唯一的快速验证类别。"""
    matched_classes = [
        class_name
        for class_name, prefix in class_prefixes.items()
        if filename.startswith(f"{prefix}_")
    ]
    if len(matched_classes) != 1:
        raise ValueError(
            f"输出文件 {filename} 应唯一匹配一个快速验证类别，实际匹配: {matched_classes}"
        )
    return matched_classes[0]


def macro_average_class_metrics(class_totals, expected_classes=QUICK_VAL_CLASS_NAMES):
    """先计算每类指标均值，再对所有指定类别进行等权宏平均。"""
    missing_classes = [name for name in expected_classes if name not in class_totals]
    if missing_classes:
        raise ValueError(f"缺少快速验证类别统计: {missing_classes}")

    per_class_metrics = {}
    for class_name in expected_classes:
        totals = class_totals[class_name]
        count = int(totals['count'])
        if count <= 0:
            raise ValueError(f"快速验证类别 {class_name} 没有可用样本")

        per_class_metrics[class_name] = {
            'count': count,
            'psnr': totals['psnr'] / count,
            'ssim': totals['ssim'] / count,
            'lpips': totals['lpips'] / count,
        }

    class_count = len(expected_classes)
    macro_metrics = (
        sum(per_class_metrics[name]['psnr'] for name in expected_classes) / class_count,
        sum(per_class_metrics[name]['ssim'] for name in expected_classes) / class_count,
        sum(per_class_metrics[name]['lpips'] for name in expected_classes) / class_count,
    )
    return macro_metrics, per_class_metrics


def summarize_best_metrics(eval_epochs, psnr_values, ssim_values, lpips_values):
    """安全汇总最佳指标；没有评估记录时返回 None，不调用 max/min。"""
    lengths = {
        len(eval_epochs), len(psnr_values), len(ssim_values), len(lpips_values)
    }
    if len(lengths) != 1:
        raise ValueError("评估 epoch 与 PSNR/SSIM/LPIPS 记录长度不一致")
    if not psnr_values:
        return None

    best_psnr_index = psnr_values.index(max(psnr_values))
    best_ssim_index = ssim_values.index(max(ssim_values))
    best_lpips_index = lpips_values.index(min(lpips_values))
    return {
        'psnr': {
            'index': best_psnr_index,
            'value': psnr_values[best_psnr_index],
            'epoch': eval_epochs[best_psnr_index],
        },
        'ssim': {
            'index': best_ssim_index,
            'value': ssim_values[best_ssim_index],
            'epoch': eval_epochs[best_ssim_index],
        },
        'lpips': {
            'index': best_lpips_index,
            'value': lpips_values[best_lpips_index],
            'epoch': eval_epochs[best_lpips_index],
        },
    }
