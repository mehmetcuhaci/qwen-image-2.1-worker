"""Startup-only timings. Hooks are restored before any inference is accepted.

Nested/parallel call totals are CPU wall time, not additive wall-clock stages.
Tensor transfers can be asynchronous; drain time is reported separately.
"""
from contextlib import contextmanager
from functools import wraps
import logging
from pathlib import Path
import threading
import time


def memory():
    values = {}
    try:
        for line in Path('/proc/self/status').read_text().splitlines():
            key, _, value = line.partition(':')
            if key in {'VmRSS', 'VmHWM', 'RssAnon', 'RssFile'}:
                values[key + '_bytes'] = int(value.split()[0]) * 1024
    except OSError:
        pass
    return values


class StartupProfile:
    def __init__(self):
        self.stages = {}
        self.calls = {}
        self.lock = threading.Lock()
        self.before = memory()
        self.load_options = {}

    def record(self, name, elapsed, size=0):
        with self.lock:
            item = self.calls.setdefault(name, {'calls': 0, 'host_ms': 0, 'bytes': 0})
            item['calls'] += 1
            item['host_ms'] += elapsed * 1000
            item['bytes'] += size

    @contextmanager
    def stage(self, name):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = round((time.perf_counter() - start) * 1000, 3)

    def report(self):
        return {'stages_ms': self.stages.copy(), 'calls': {
            name: dict(item, host_ms=round(item['host_ms'], 3))
            for name, item in sorted(self.calls.items())},
            'memory_before': self.before, 'memory_after': memory(),
            'load_options': self.load_options.copy(),
            'timing_note': 'Call timings overlap and measure host time; CUDA drain is separate.'}

    @contextmanager
    def loading_hooks(self, torch):
        # Scope these wrappers to startup. No CUDA sync per tensor, no profiler,
        # warm-up inference, background sampler, or retained tensor references.
        from diffusers.models import model_loading_utils, modeling_utils
        from diffusers.pipelines import pipeline_utils
        restores = []

        def patch(owner, name, replacement):
            original = getattr(owner, name)
            restores.append((owner, name, original))
            setattr(owner, name, replacement(original))

        def timed(label):
            def replace(original):
                @wraps(original)
                def wrapper(*args, **kwargs):
                    start = time.perf_counter()
                    try:
                        return original(*args, **kwargs)
                    finally:
                        self.record(label, time.perf_counter() - start)
                return wrapper
            return replace

        def component(original):
            @wraps(original)
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                name = kwargs.get('name', args[14] if len(args) > 14 else 'unknown')
                self.load_options[name] = {key: str(kwargs[key]) for key in
                    ('dtype', 'device_map', 'low_cpu_mem_usage', 'use_safetensors', 'disable_mmap')
                    if key in kwargs}
                try:
                    return original(*args, **kwargs)
                finally:
                    self.record('component.' + name, time.perf_counter() - start)
            return wrapper

        def tensor_to(original):
            @wraps(original)
            def wrapper(tensor, *args, **kwargs):
                start = time.perf_counter()
                output = original(tensor, *args, **kwargs)
                elapsed = time.perf_counter() - start
                if tensor.device != output.device:
                    label = 'placement.' + tensor.device.type + '_to_' + output.device.type
                elif tensor.dtype != output.dtype:
                    label = 'dtype.' + str(tensor.dtype) + '_to_' + str(output.dtype)
                elif tensor is not output:
                    label = 'copy.' + tensor.device.type
                else:
                    return output
                self.record(label, elapsed, tensor.numel() * tensor.element_size())
                return output
            return wrapper

        try:
            patch(pipeline_utils, 'load_sub_model', component)
            # Diffusers imports aliases, so cover both actual call sites.
            for module in (model_loading_utils, modeling_utils):
                patch(module, 'load_state_dict', timed('checkpoint.diffusers.read'))
            patch(model_loading_utils, 'load_model_dict_into_meta', timed('checkpoint.diffusers.install'))
            patch(torch, 'load', timed('checkpoint.torch.read'))
            if hasattr(torch, 'nn'):
                patch(torch.nn.Module, 'load_state_dict', timed('checkpoint.torch.install'))
            patch(torch.Tensor, 'to', tensor_to)
            yield
        finally:
            for owner, name, original in reversed(restores):
                setattr(owner, name, original)


def component_inventory(pipe):
    result = {}
    for name, model in pipe.components.items():
        if not hasattr(model, 'parameters'):
            continue
        counts = {}
        for tensor in model.parameters():
            key = str(tensor.device) + '/' + str(tensor.dtype)
            counts[key] = counts.get(key, 0) + tensor.numel() * tensor.element_size()
        result[name] = counts
    return result
