"""DA-MambaNet 命令行参数、服务器兼容范围与消融配置回归测试。"""

import contextlib
import io
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.options import option
from net.DA_MambaNet import DA_MambaNet


class _FailIfCalled(nn.Module):
    """在前向传播被调用时立即失败，用于确认 DAM 已被真正旁路。"""

    def forward(self, _x):
        """阻止测试中的禁用模块被意外执行。"""
        raise AssertionError("use_dam=False 时不应调用 DAM")


class TestBooleanOptions(unittest.TestCase):
    """验证命令行中的显式真假字符串可以被正确解析。"""

    def test_false_and_true_strings(self):
        """确认 False/0/No 与 True/1/Yes 均得到预期布尔值。"""
        args = option().parse_args([
            '--use_dam', 'False',
            '--use_film', '0',
            '--gpu_mode', 'No',
            '--shuffle', 'True',
            '--allinone', '1',
            '--grad_clip', 'Yes',
        ])

        self.assertFalse(args.use_dam)
        self.assertFalse(args.use_film)
        self.assertFalse(args.gpu_mode)
        self.assertTrue(args.shuffle)
        self.assertTrue(args.allinone)
        self.assertTrue(args.grad_clip)


class TestDConvOptions(unittest.TestCase):
    """验证 d_conv 默认值和服务器 CUDA 内核兼容范围。"""

    def test_default_is_four(self):
        """确认未传参时使用 causal_conv1d 兼容的默认宽度4。"""
        args = option().parse_args([])
        self.assertEqual(args.d_conv, 4)

    def test_widths_two_to_four_are_allowed(self):
        """确认敏感性实验允许使用宽度2、3、4。"""
        for width in (2, 3, 4):
            with self.subTest(width=width):
                args = option().parse_args(['--d_conv', str(width)])
                self.assertEqual(args.d_conv, width)

    def test_width_five_is_rejected_before_model_creation(self):
        """确认不兼容宽度5会在命令行解析阶段被拒绝。"""
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                option().parse_args(['--d_conv', '5'])


class TestAblationPropagation(unittest.TestCase):
    """验证消融配置能够传播到全部 CMB 并改变真实计算路径。"""

    @staticmethod
    def _build_model(**kwargs):
        """构建小通道模型，以较低开销完成配置级测试。"""
        return DA_MambaNet(
            channels=[8, 8, 16, 32],
            num_classes=5,
            d_state=4,
            d_conv=2,
            expand=2,
            use_rgb_refiner=False,
            **kwargs,
        )

    def test_scan_mode_and_film_reach_every_cmb(self):
        """确认三种扫描模式和 FiLM 开关会传递到12个CMB。"""
        expected_scans = {
            'hetero': (2, 4),
            'all_2way': (2, 2),
            'all_4way': (4, 4),
        }

        for scan_mode, (hv_scan, i_scan) in expected_scans.items():
            with self.subTest(scan_mode=scan_mode):
                model = self._build_model(
                    use_dam=True,
                    use_film=False,
                    scan_mode=scan_mode,
                )
                for index in range(1, 7):
                    hv_cmb = getattr(model, f'HV_CMB{index}')
                    i_cmb = getattr(model, f'I_CMB{index}')
                    self.assertFalse(hv_cmb.use_film)
                    self.assertFalse(i_cmb.use_film)
                    self.assertEqual(hv_cmb.ss2d.num_scan, hv_scan)
                    self.assertEqual(i_cmb.ss2d.num_scan, i_scan)

    def test_disabled_dam_is_bypassed_and_returns_zero_condition(self):
        """确认关闭 DAM 后不执行该模块，并返回全零条件向量。"""
        model = self._build_model(
            use_dam=False,
            use_film=False,
            scan_mode='all_2way',
        )
        model.dam = _FailIfCalled()
        model.eval()

        input_tensor = torch.rand(1, 3, 8, 8)
        with torch.no_grad():
            output, condition = model(input_tensor)

        self.assertEqual(output.shape, input_tensor.shape)
        self.assertEqual(condition.shape, (1, 6))
        self.assertTrue(torch.equal(condition, torch.zeros_like(condition)))

    def test_invalid_scan_mode_is_rejected(self):
        """确认未知扫描模式会立即给出明确错误。"""
        with self.assertRaises(ValueError):
            self._build_model(
                use_dam=True,
                use_film=True,
                scan_mode='invalid',
            )

    def test_dam_returns_raw_logits_for_classification_loss(self):
        """确认分类损失拿到原始 logits，而条件向量仍保留概率。"""
        model = self._build_model(
            use_dam=True,
            use_film=False,
            scan_mode='all_2way',
        )
        model.eval()

        input_tensor = torch.rand(2, 3, 8, 8)
        with torch.no_grad():
            output, condition, logits = model(
                input_tensor, return_dam_logits=True
            )

        self.assertEqual(output.shape, input_tensor.shape)
        self.assertEqual(logits.shape, (2, 5))
        self.assertTrue(torch.allclose(
            condition[:, :5], F.softmax(logits, dim=-1), atol=1e-6
        ))


if __name__ == '__main__':
    unittest.main()
