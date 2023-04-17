import datetime
import sys
from unittest import TestCase

from mmengine import DefaultScope

from lqit.utils import register_all_modules


class TestSetupEnv(TestCase):

    def test_register_all_modules(self):
        from lqit.registry import DATASETS

        # not init default scope
        sys.modules.pop('lqit.edit.datasets', None)
        sys.modules.pop('lqit.edit.datasets.basic_image_dataset', None)
        DATASETS._module_dict.pop('BasicImageDataset', None)
        self.assertFalse('BasicImageDataset' in DATASETS.module_dict)
        register_all_modules(init_default_scope=False)
        self.assertTrue('BasicImageDataset' in DATASETS.module_dict)

        # init default scope
        sys.modules.pop('lqit.edit.datasets')
        sys.modules.pop('lqit.edit.datasets.basic_image_dataset')
        DATASETS._module_dict.pop('BasicImageDataset', None)
        self.assertFalse('BasicImageDataset' in DATASETS.module_dict)
        register_all_modules(init_default_scope=True)
        self.assertTrue('BasicImageDataset' in DATASETS.module_dict)
        self.assertEqual(DefaultScope.get_current_instance().scope_name,
                         'lqit')

        # init default scope when another scope is init
        name = f'test-{datetime.datetime.now()}'
        DefaultScope.get_instance(name, scope_name='test')
        with self.assertWarnsRegex(
                Warning, 'The current default scope "test" is not "lqit"'):
            register_all_modules(init_default_scope=True)
