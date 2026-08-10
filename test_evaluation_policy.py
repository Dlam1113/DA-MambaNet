"""快速验证宏平均和最终评估调度的回归测试。"""

import unittest

from evaluation_utils import (
    macro_average_class_metrics,
    resolve_quick_val_class,
    should_run_evaluation,
    summarize_best_metrics,
)


class TestEvaluationSchedule(unittest.TestCase):
    """验证训练末尾一定评估，同时保留正常周期评估。"""

    def test_short_training_still_evaluates_last_epoch(self):
        """确认 nEpochs 小于 snapshots 时只在最终 epoch 执行评估。"""
        decisions = [should_run_evaluation(epoch, 5, 30) for epoch in range(1, 6)]
        self.assertEqual(decisions, [False, False, False, False, True])

    def test_periodic_and_final_epochs_are_evaluated(self):
        """确认周期节点和非周期的最终节点都会执行评估。"""
        eval_epochs = [
            epoch for epoch in range(1, 66)
            if should_run_evaluation(epoch, 65, 30)
        ]
        self.assertEqual(eval_epochs, [30, 60, 65])

    def test_invalid_snapshot_interval_is_rejected(self):
        """确认无效评估周期会得到明确错误。"""
        with self.assertRaises(ValueError):
            should_run_evaluation(1, 5, 0)


class TestMacroAverage(unittest.TestCase):
    """验证类别样本数不同时仍按类别等权汇总。"""

    def test_classes_are_averaged_with_equal_weight(self):
        """确认宏平均等于五个类别均值的算术平均，而非图片级平均。"""
        class_means = {
            'lowlight': (15, 10.0, 0.50, 0.50),
            'fog': (30, 20.0, 0.60, 0.40),
            'rain': (30, 30.0, 0.70, 0.30),
            'snow': (30, 40.0, 0.80, 0.20),
            'blur': (30, 50.0, 0.90, 0.10),
        }
        totals = {
            name: {
                'count': count,
                'psnr': psnr * count,
                'ssim': ssim * count,
                'lpips': lpips * count,
            }
            for name, (count, psnr, ssim, lpips) in class_means.items()
        }

        macro_metrics, per_class = macro_average_class_metrics(totals)

        self.assertAlmostEqual(macro_metrics[0], 30.0)
        self.assertAlmostEqual(macro_metrics[1], 0.70)
        self.assertAlmostEqual(macro_metrics[2], 0.30)
        self.assertEqual(per_class['lowlight']['count'], 15)
        self.assertEqual(per_class['fog']['count'], 30)

    def test_missing_class_is_rejected(self):
        """确认缺少任一类别时不会静默计算伪五类均值。"""
        with self.assertRaises(ValueError):
            macro_average_class_metrics({})


class TestClassPrefixResolution(unittest.TestCase):
    """验证预测文件能够稳定映射到唯一的快速验证类别。"""

    def test_unique_prefix_is_resolved(self):
        """确认带数据集前缀的文件名能映射到正确类别。"""
        prefixes = {'lowlight': 'eval15', 'fog': 'Fog_val'}
        self.assertEqual(
            resolve_quick_val_class('Fog_val_scene_001.png', prefixes),
            'fog',
        )

    def test_unmatched_or_ambiguous_prefix_is_rejected(self):
        """确认未匹配或同时匹配多个前缀时会立即报错。"""
        with self.assertRaises(ValueError):
            resolve_quick_val_class('unknown.png', {'fog': 'Fog_val'})
        with self.assertRaises(ValueError):
            resolve_quick_val_class(
                'Fog_val_scene.png',
                {'fog': 'Fog', 'rain': 'Fog_val'},
            )


class TestBestMetricSummary(unittest.TestCase):
    """验证空评估记录安全处理和真实 epoch 映射。"""

    def test_empty_history_returns_none(self):
        """确认空列表不会触发 max/min，而是返回无结果。"""
        self.assertIsNone(summarize_best_metrics([], [], [], []))

    def test_best_values_use_actual_evaluation_epochs(self):
        """确认最终非周期评估使用真实 epoch，而不是按下标推算。"""
        summary = summarize_best_metrics(
            [30, 60, 65],
            [20.0, 22.0, 21.0],
            [0.70, 0.72, 0.71],
            [0.30, 0.25, 0.20],
        )
        self.assertEqual(summary['psnr']['epoch'], 60)
        self.assertEqual(summary['ssim']['epoch'], 60)
        self.assertEqual(summary['lpips']['epoch'], 65)


if __name__ == '__main__':
    unittest.main()
