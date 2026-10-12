import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import pytest

spec = importlib.util.spec_from_file_location('startup', Path(__file__).parents[1] / 'qwen_worker/startup.py')
startup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(startup)


@pytest.mark.parametrize('fails', [False, True])
def test_startup_hooks_restore_and_keep_original_arguments(monkeypatch, fails):
    original = lambda *args, **kwargs: kwargs
    modules = {}
    for name in ['diffusers', 'diffusers.models', 'diffusers.models.modeling_utils',
                 'diffusers.models.model_loading_utils', 'diffusers.pipelines',
                 'diffusers.pipelines.pipeline_utils']:
        modules[name] = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, modules[name])
    for name in ['diffusers.models.modeling_utils', 'diffusers.models.model_loading_utils']:
        modules[name].load_state_dict = original
    modules['diffusers.models.model_loading_utils'].load_model_dict_into_meta = original
    pipeline = modules['diffusers.pipelines.pipeline_utils']
    pipeline.load_sub_model = original
    class Tensor:
        to = original
    torch = SimpleNamespace(load=original, Tensor=Tensor)
    profile = startup.StartupProfile()
    try:
        with profile.loading_hooks(torch):
            assert pipeline.load_sub_model(name='transformer', dtype='bf16') == {'name':'transformer','dtype':'bf16'}
            if fails:
                raise ValueError('failed loading')
    except ValueError:
        pass
    assert pipeline.load_sub_model is original
    assert torch.load is original and Tensor.to is original
    assert profile.report()['calls']['component.transformer']['calls'] == 1


def test_tensor_noop_is_not_counted_as_ram_copy(monkeypatch):
    modules = {}
    for name in ['diffusers', 'diffusers.models', 'diffusers.models.modeling_utils',
                 'diffusers.models.model_loading_utils', 'diffusers.pipelines',
                 'diffusers.pipelines.pipeline_utils']:
        modules[name] = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, modules[name])
    original = lambda *args, **kwargs: None
    for name in ['diffusers.models.modeling_utils', 'diffusers.models.model_loading_utils']:
        modules[name].load_state_dict = original
    modules['diffusers.models.model_loading_utils'].load_model_dict_into_meta = original
    modules['diffusers.pipelines.pipeline_utils'].load_sub_model = original
    class Tensor:
        def __init__(self, dtype):
            self.dtype, self.device = dtype, SimpleNamespace(type='cpu')
        def to(self, dtype):
            if dtype == self.dtype:
                return self
            output = Tensor(dtype)
            output.device = self.device
            return output
        def numel(self):
            return 8
        def element_size(self):
            return 2
    torch = SimpleNamespace(load=original, Tensor=Tensor)
    profile = startup.StartupProfile()
    with profile.loading_hooks(torch):
        value = Tensor('torch.bfloat16')
        assert value.to('torch.bfloat16') is value
        other = value.to('torch.float32')
        assert other.dtype == 'torch.float32'
    calls = profile.report()['calls']
    assert calls['dtype.torch.bfloat16_to_torch.float32']['bytes'] == 16
    assert 'copy.cpu' not in calls
