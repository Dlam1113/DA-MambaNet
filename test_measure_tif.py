"""AllInOne 快速验证 TIF/TIFF 雪图统计回归测试。"""

import os
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from measure import discover_metric_images, metrics, resolve_gt_image_path


class _FakeCudaTensor:
    """提供测试所需的最小 CUDA 兼容接口，不执行真实 GPU 计算。"""

    def cuda(self):
        """模拟张量迁移到 CUDA，并原样返回自身。"""
        return self


class _FakeScore:
    """提供 LPIPS 返回值所需的 item 接口。"""

    def item(self):
        """返回固定的零感知距离。"""
        return 0.0


class _FakeLPIPS:
    """替代真实 LPIPS 网络，使文件发现测试不依赖 GPU。"""

    def cuda(self):
        """模拟模型迁移到 CUDA，并原样返回自身。"""
        return self

    def forward(self, _reference, _prediction):
        """返回固定 LPIPS 分数，隔离与本问题无关的网络计算。"""
        return _FakeScore()


class TestTifQuickValidation(unittest.TestCase):
    """验证 TIF/TIFF 雪图能进入五类快速验证统计。"""

    @staticmethod
    def _save_image(path, value):
        """生成可由 PIL 读取的16×16 RGB测试图。"""
        image = np.full((16, 16, 3), value, dtype=np.uint8)
        Image.fromarray(image).save(path)

    def test_all_tif_extension_variants_are_discovered_and_matched(self):
        """确认四种 TIF/TIFF 大小写组合均可发现并跨扩展名匹配 GT。"""
        with tempfile.TemporaryDirectory() as prediction_dir, tempfile.TemporaryDirectory() as gt_dir:
            prediction_extensions = ('.tif', '.tiff', '.TIF', '.TIFF')
            gt_extensions = ('.TIFF', '.TIF', '.tiff', '.tif')
            expected_predictions = []

            for index, (prediction_ext, gt_ext) in enumerate(
                zip(prediction_extensions, gt_extensions), start=1
            ):
                stem = f'CSD_scene_{index:03d}'
                prediction_path = os.path.join(prediction_dir, stem + prediction_ext)
                gt_path = os.path.join(gt_dir, stem + gt_ext)
                self._save_image(prediction_path, 128)
                self._save_image(gt_path, 128)
                expected_predictions.append(prediction_path)

                self.assertEqual(
                    resolve_gt_image_path(gt_dir, os.path.basename(prediction_path)),
                    gt_path,
                )

            self.assertEqual(
                discover_metric_images(prediction_dir),
                sorted(expected_predictions),
            )

    @mock.patch('measure.torch.cuda.empty_cache')
    @mock.patch('measure.lpips.im2tensor', return_value=_FakeCudaTensor())
    @mock.patch('measure.lpips.LPIPS', return_value=_FakeLPIPS())
    def test_tif_snow_is_included_in_five_class_macro_average(
        self, _lpips_model, _im2tensor, _empty_cache
    ):
        """确认 TIF 雪图计数为1，且五类等权宏平均路径保持可用。"""
        class_prefixes = {
            'lowlight': 'eval15',
            'fog': 'Fog_val',
            'rain': 'Rain_val',
            'snow': 'CSD_val',
            'blur': 'GoPro_val',
        }
        filenames = {
            'lowlight': 'eval15_scene_001.png',
            'fog': 'Fog_val_scene_001.jpg',
            'rain': 'Rain_val_scene_001.PNG',
            'snow': 'CSD_val_scene_001.TIF',
            'blur': 'GoPro_val_scene_001.jpeg',
        }

        with tempfile.TemporaryDirectory() as prediction_dir, tempfile.TemporaryDirectory() as gt_dir:
            for index, (class_name, filename) in enumerate(filenames.items(), start=1):
                prediction_path = os.path.join(prediction_dir, filename)
                self._save_image(prediction_path, index * 20)

                gt_filename = filename
                if class_name == 'snow':
                    gt_filename = 'CSD_val_scene_001.tiff'
                self._save_image(os.path.join(gt_dir, gt_filename), index * 20)

            _psnr, _ssim, _lpips, per_class = metrics(
                prediction_dir,
                gt_dir,
                use_GT_mean=False,
                class_prefixes=class_prefixes,
            )

            self.assertEqual(per_class['snow']['count'], 1)
            self.assertEqual(
                [per_class[name]['count'] for name in class_prefixes],
                [1, 1, 1, 1, 1],
            )


if __name__ == '__main__':
    unittest.main()
