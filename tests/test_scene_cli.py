"""Native launcher regression: project paths can contain spaces on macOS."""

import importlib.util
from pathlib import Path

import pytest


def _script():
    path = Path(__file__).resolve().parents[1] / 'scripts/scene.py'
    spec = importlib.util.spec_from_file_location('scene_script', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mac_launcher_invokes_python_trampoline_even_with_spaces(tmp_path, monkeypatch):
    module = _script()
    directory = tmp_path / 'Project With Spaces/.venv/bin'
    directory.mkdir(parents=True)
    launcher = directory / 'mjpython'
    launcher.write_text('#!invalid interpreter with spaces\n')
    executable = str(directory / 'python')
    monkeypatch.setattr(module.sys, 'platform', 'darwin')
    monkeypatch.setattr(module.sys, 'executable', executable)
    monkeypatch.setattr(module.sys, 'argv', ['scene.py', 'view', '--scene', 'room'])
    monkeypatch.delenv('MJPYTHON_BIN', raising=False)
    calls = []
    monkeypatch.setattr(module.os, 'execve', lambda *args: calls.append(args))
    module._mac_view_runtime()
    target, args, env = calls[0]
    assert target == executable
    assert args[:2] == [executable, str(launcher)]
    assert args[-3:] == ['view', '--scene', 'room']
    monkeypatch.setenv('MJPYTHON_BIN', '/native/runtime')
    module._mac_view_runtime()
    assert len(calls) == 1
