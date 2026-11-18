"""PACE result downloads preserve literal paths, local files, and failures."""

import json
import os
from pathlib import Path
import pty
import select
import shlex
import shutil
import subprocess
import sys
import time

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/pace/download_from_pace.sh'
PROGRESS = SCRIPT.with_name('rsync_progress.awk')
REAL_RSYNC = shutil.which('rsync')


@pytest.fixture
def download_project(tmp_path):
    project = tmp_path / 'local project with spaces'
    project.mkdir()
    (project / 'pyproject.toml').write_text('[project]\nname="fixture"\n')
    remote = tmp_path / 'fake remote'
    remote.mkdir()
    record = tmp_path / 'invocation.json'
    fake_bin = tmp_path / 'bin'
    fake_bin.mkdir()
    implementation = tmp_path / 'fake_rsync.py'
    implementation.write_text('''
import json, os, subprocess, sys, time
from pathlib import Path
args = sys.argv[1:]
if '--version' in args:
    Path(os.environ['TEST_INVOCATION']).with_suffix('.probe.json').write_text(json.dumps(args))
    if os.environ.get('TEST_LOCAL_COPY') == '1':
        environment = {**os.environ, 'PATH': os.environ['TEST_BASE_PATH']}
        sys.exit(subprocess.call([os.environ['TEST_REAL_RSYNC'], *args], env=environment))
    sys.exit(0 if os.environ.get('TEST_PROGRESS2_SUPPORTED') == '1' else 1)
Path(os.environ['TEST_INVOCATION']).write_text(json.dumps({
    'args': args, 'cwd': str(Path.cwd())}))
if os.environ.get('TEST_LOCAL_COPY') == '1':
    source = args[-2].split(':', 1)[1]
    args[-2] = str(Path(os.environ['TEST_REMOTE_ROOT']) / source) + '/'
    environment = {**os.environ, 'PATH': os.environ['TEST_BASE_PATH']}
    sys.exit(subprocess.call([os.environ['TEST_REAL_RSYNC'], *args], env=environment))
sys.stdout.write(os.environ.get('TEST_RSYNC_OUTPUT', ''))
sys.stdout.flush()
if os.environ.get('TEST_STREAM_RELEASE'):
    deadline = time.monotonic() + 5
    while not Path(os.environ['TEST_STREAM_RELEASE']).exists() and time.monotonic() < deadline:
        time.sleep(0.01)
sys.exit(int(os.environ.get('TEST_EXIT_CODE', '0')))
''')
    rsync = fake_bin / 'rsync'
    rsync.write_text('#!/usr/bin/env bash\nexec ' + shlex.quote(sys.executable) + ' ' +
                     shlex.quote(str(implementation)) + ' "$@"\n')
    rsync.chmod(0o755)
    ssh = fake_bin / 'ssh'
    ssh.write_text('#!/usr/bin/env bash\nprintf "SSH must not run in tests" >&2\nexit 99\n')
    ssh.chmod(0o755)
    environment = os.environ.copy()
    for name in ('WTUMI_PACE_HOST', 'WTUMI_PACE_DIR', 'WTUMI_DOWNLOAD_ROOT',
                 'WTUMI_WORKDIR', 'TEST_LOCAL_COPY', 'TEST_EXIT_CODE',
                 'TEST_PROGRESS2_SUPPORTED', 'TEST_RSYNC_OUTPUT', 'TEST_STREAM_RELEASE'):
        environment.pop(name, None)
    environment.update(WTUMI_WORKDIR=str(project), TEST_INVOCATION=str(record),
                       TEST_REMOTE_ROOT=str(remote), TEST_REAL_RSYNC=REAL_RSYNC or '',
                       TEST_BASE_PATH=environment.get('PATH', ''),
                       PATH=str(fake_bin) + os.pathsep + environment.get('PATH', ''))
    return project, remote, record, environment


def _run(environment, *args, cwd=None):
    return subprocess.run(['bash', str(SCRIPT), *args], env=environment, cwd=cwd,
                          capture_output=True, text=True)


def _run_on_terminal(environment, *args):
    """Exercise the stdout-TTY branch without invoking any remote command."""
    master, slave = pty.openpty()
    try:
        result = subprocess.run(['bash', str(SCRIPT), *args], env=environment,
                                stdout=slave, stderr=subprocess.PIPE, timeout=10)
        os.set_blocking(master, False)
        chunks = []
        while True:
            try:
                chunk = os.read(master, 65536)
            except OSError:
                break
            if not chunk:
                break
            chunks.append(chunk)
        return result.returncode, b''.join(chunks).decode(), result.stderr.decode()
    finally:
        os.close(master)
        os.close(slave)


def test_default_download_resolves_project_from_another_working_directory(download_project):
    project, remote, record, environment = download_project
    result = _run(environment, 'g1_vision_30fps_pace', cwd=remote)
    assert result.returncode == 0, result.stderr
    args = json.loads(record.read_text())['args']
    assert args[-2:] == [
        'djiang94@login-ice.pace.gatech.edu:projects/WT-UMI-2_rgbd2voxel/outputs/g1_vision_30fps_pace/',
        str(project / 'outputs/g1_vision_30fps_pace') + '/']
    assert {'-az', '-v', '--progress', '--partial-dir=.rsync-partial', '--safe-links'} <= set(args)
    assert '--delete' not in args
    assert (project / 'outputs/g1_vision_30fps_pace').is_dir()
    assert not (remote / 'outputs').exists()
    assert 'Download completed:' in result.stdout


def test_literal_relative_root_is_created_under_project(download_project):
    project, remote, record, environment = download_project
    root = 'downloads with spaces/$(touch unexpected)'
    result = _run(environment, 'run_01', '--local-root', root, cwd=remote)
    assert result.returncode == 0, result.stderr
    assert json.loads(record.read_text())['args'][-1] == str(project / root / 'run_01') + '/'
    assert (project / root / 'run_01').is_dir()
    assert not (project / 'unexpected').exists() and not (remote / 'unexpected').exists()


def test_environment_defaults_and_explicit_overrides(download_project):
    project, remote, record, environment = download_project
    environment.update(WTUMI_PACE_HOST='pace-alias', WTUMI_PACE_DIR='/scratch/user/my-project/',
                       WTUMI_DOWNLOAD_ROOT='downloaded')
    result = _run(environment, 'run_01')
    assert result.returncode == 0, result.stderr
    assert json.loads(record.read_text())['args'][-2:] == [
        'pace-alias:/scratch/user/my-project/outputs/run_01/', str(project / 'downloaded/run_01') + '/']
    result = _run(environment, 'run_02', '--host', 'new-user@new.host',
                  '--remote-dir', 'projects/new', '--local-root', str(remote / 'absolute root'))
    assert result.returncode == 0, result.stderr
    assert json.loads(record.read_text())['args'][-2:] == [
        'new-user@new.host:projects/new/outputs/run_02/', str(remote / 'absolute root/run_02') + '/']


def test_list_does_not_create_local_result_directories(download_project):
    project, _, record, environment = download_project
    result = _run(environment, '--list')
    assert result.returncode == 0, result.stderr
    assert json.loads(record.read_text())['args'] == [
        '--list-only', 'djiang94@login-ice.pace.gatech.edu:projects/WT-UMI-2_rgbd2voxel/outputs/']
    assert not (project / 'outputs').exists()
    assert not record.with_suffix('.probe.json').exists()


@pytest.mark.parametrize('args', [('run_01', '--dry-run'), ('--list', '--dry-run')])
def test_dry_run_is_preview_only_without_transfer_or_directory_creation(download_project, args):
    project, _, record, environment = download_project
    result = _run(environment, *args)
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout.splitlines()[-1])
    assert command[0] == 'rsync'
    assert not record.exists() and not (project / 'outputs').exists()
    assert 'Download completed:' not in result.stdout


@pytest.mark.parametrize('supports_progress2', [False, True])
def test_progress_flags_follow_local_rsync_capabilities(download_project, supports_progress2):
    _, _, record, environment = download_project
    environment['TEST_PROGRESS2_SUPPORTED'] = '1' if supports_progress2 else '0'
    result = _run(environment, 'run_01')
    assert result.returncode == 0, result.stderr
    probe = json.loads(record.with_suffix('.probe.json').read_text())
    assert set(probe) == {'--info=progress2', '--no-inc-recursive', '--version'}
    args = json.loads(record.read_text())['args']
    assert ('--info=progress2' in args) == supports_progress2
    assert ('--no-inc-recursive' in args) == supports_progress2
    assert ('--progress' in args) != supports_progress2


@pytest.mark.parametrize('terminal', [False, True])
def test_no_progress_disables_probe_flags_and_graphical_bar(download_project, terminal):
    _, _, record, environment = download_project
    environment.update(TEST_PROGRESS2_SUPPORTED='1',
                       TEST_RSYNC_OUTPUT='output-file.mp4\n  512 50% 2.00MB/s 0:00:01\r')
    if terminal:
        code, stdout, stderr = _run_on_terminal(environment, 'run_01', '--no-progress')
    else:
        result = _run(environment, 'run_01', '--no-progress')
        code, stdout, stderr = result.returncode, result.stdout, result.stderr
    assert code == 0, stderr
    assert not record.with_suffix('.probe.json').exists()
    args = json.loads(record.read_text())['args']
    assert not {'--progress', '--info=progress2', '--no-inc-recursive'} & set(args)
    assert '[' not in stdout
    assert '512 50% 2.00MB/s 0:00:01' in stdout


def test_non_terminal_output_keeps_original_rsync_progress(download_project):
    _, _, _, environment = download_project
    raw_output = 'comparison.mp4\n  512 50% 2.00MB/s 0:00:01\r  1024 100% 2.00MB/s 0:00:00\n'
    result = _run({**environment, 'TEST_RSYNC_OUTPUT': raw_output}, 'run_01')
    assert result.returncode == 0, result.stderr
    assert raw_output.replace('\r', '\n') in result.stdout
    assert '[' not in result.stdout


@pytest.mark.parametrize('supports_progress2,label', [(False, 'Current file'), (True, 'Overall')])
def test_terminal_progress_bar_keeps_transfer_failure_and_partial_percentage(
        download_project, supports_progress2, label):
    _, _, _, environment = download_project
    environment.update(TEST_PROGRESS2_SUPPORTED='1' if supports_progress2 else '0',
                       TEST_RSYNC_OUTPUT='comparison.mp4\n  512 50% 2.00MB/s 0:00:01\r',
                       TEST_EXIT_CODE='23')
    code, stdout, stderr = _run_on_terminal(environment, 'run_01')
    assert code == 23, stderr
    assert label + ' [############------------]  50%' in stdout
    assert '[########################] 100%' not in stdout
    assert 'Download completed:' not in stdout
    assert 'Download did not complete' in stderr


@pytest.mark.parametrize('label', ['Current file', 'Overall'])
def test_formatter_parses_carriage_returns_newlines_and_native_progress_fields(label):
    native = ('receiving file list ... done\ncomparison.mp4\n'
              '  1,024 25% 1.00MB/s 0:00:03\r'
              '  2,048 50% 2.00MB/s 0:00:01\r'
              '  4,096 100% 2.00MB/s 0:00:00 (xfr#1, to-chk=0/1)\n'
              'total size is 4096\n')
    result = subprocess.run(['awk', '-v', 'label=' + label, '-f', str(PROGRESS)],
                            input=native, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert 'receiving file list ... done' in result.stdout
    assert 'comparison.mp4' in result.stdout and 'total size is 4096' in result.stdout
    assert label + ' [######------------------]  25%  1.00MB/s  0:00:03' in result.stdout
    assert label + ' [############------------]  50%  2.00MB/s  0:00:01' in result.stdout
    assert label + ' [########################] 100%' in result.stdout


def test_formatter_does_not_invent_completion_on_truncated_input():
    result = subprocess.run(['awk', '-f', str(PROGRESS)],
                            input='  512 50% 2.00MB/s 0:00:01',
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert '[############------------]  50%' in result.stdout
    assert '100%' not in result.stdout


def test_terminal_partial_progress_appears_before_transfer_finishes(download_project):
    project, _, _, environment = download_project
    release = project / 'release-transfer'
    environment.update(TEST_RSYNC_OUTPUT='comparison.mp4\n  512 50% 2.00MB/s 0:00:01\r',
                       TEST_STREAM_RELEASE=str(release))
    master, slave = pty.openpty()
    process = subprocess.Popen(['bash', str(SCRIPT), 'run_01'], env=environment,
                               stdout=slave, stderr=subprocess.PIPE)
    output = b''
    try:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            ready, _, _ = select.select([master], [], [], 0.1)
            if ready:
                output += os.read(master, 65536)
            if b'[############------------]  50%' in output:
                break
        assert b'[############------------]  50%' in output
        assert process.poll() is None
    finally:
        release.touch()
        _, stderr = process.communicate(timeout=10)
        os.close(master)
        os.close(slave)
    assert process.returncode == 0, stderr.decode()


@pytest.mark.parametrize('args', [
    (), ('../escape',), ('outputs/run',), ('run/',), ('run;touch unsafe',), ('run one',),
    ('run', 'another'), ('run', '--list'), ('--list', '--host', 'host;touch unsafe'),
    ('--list', '--host', '-option'), ('--list', '--host', 'host with spaces'),
    ('run', '--remote-dir', '../project'), ('run', '--remote-dir', 'project/../other'),
    ('run', '--remote-dir', '/scratch/./project'), ('run', '--remote-dir', '/'),
    ('run', '--remote-dir', 'projects/$(touch unsafe)'), ('--host',), ('--unknown',),
])
def test_invalid_arguments_fail_before_transfer_or_directory_creation(download_project, args):
    project, _, record, environment = download_project
    result = _run(environment, *args)
    assert result.returncode == 2, result.stderr
    assert not record.exists() and not (project / 'outputs').exists()


@pytest.mark.parametrize('kind', ['file', 'directory_symlink', 'dangling_symlink'])
def test_invalid_destination_is_preserved(download_project, kind):
    project, remote, record, environment = download_project
    destination = project / 'outputs/run_01'
    destination.parent.mkdir()
    if kind == 'file':
        destination.write_text('keep local data')
    else:
        destination.symlink_to(remote if kind == 'directory_symlink' else project / 'missing')
    result = _run(environment, 'run_01')
    assert result.returncode == 2 and not record.exists()
    assert 'not a file or symlink' in result.stderr
    if kind == 'file':
        assert destination.read_text() == 'keep local data'
    else:
        assert destination.is_symlink()


@pytest.mark.parametrize('args', [('run_01',), ('--list',)])
def test_transfer_error_is_preserved_and_not_reported_as_success(download_project, args):
    _, _, record, environment = download_project
    result = _run({**environment, 'TEST_EXIT_CODE': '23'}, *args)
    assert result.returncode == 23
    assert record.is_file() and 'Download completed:' not in result.stdout


@pytest.mark.skipif(REAL_RSYNC is None, reason='A local rsync executable is required')
def test_real_local_rsync_renders_terminal_bar_without_network(download_project):
    project, remote, _, environment = download_project
    source = remote / 'projects/WT-UMI-2_rgbd2voxel/outputs/run_01'
    source.mkdir(parents=True)
    payload = b'local transfer fixture\n' * 1024
    (source / 'comparison.mp4').write_bytes(payload)
    environment['TEST_LOCAL_COPY'] = '1'
    code, stdout, stderr = _run_on_terminal(environment, 'run_01')
    assert code == 0, stderr
    assert (project / 'outputs/run_01/comparison.mp4').read_bytes() == payload
    assert '[########################] 100%' in stdout
    assert 'Download completed:' in stdout


@pytest.mark.skipif(REAL_RSYNC is None, reason='A local rsync executable is required')
def test_repeated_local_transfer_updates_remote_content_and_preserves_local_only_files(download_project):
    project, remote, _, environment = download_project
    source = remote / 'projects/WT-UMI-2_rgbd2voxel/outputs/run_01'
    (source / 'room').mkdir(parents=True)
    (source / 'room/comparison.mp4').write_bytes(b'initial video')
    (source / 'index.html').write_text('gallery')
    environment['TEST_LOCAL_COPY'] = '1'
    first = _run(environment, 'run_01')
    assert first.returncode == 0, first.stderr
    destination = project / 'outputs/run_01'
    assert (destination / 'room/comparison.mp4').read_bytes() == b'initial video'
    assert 'Open the result gallery on the Mac:' in first.stdout
    (destination / 'local_notes.txt').write_text('keep my notes')
    (source / 'room/comparison.mp4').write_bytes(b'updated video with more frames')
    (source / 'index.html').unlink()
    second = _run(environment, 'run_01')
    assert second.returncode == 0, second.stderr
    assert (destination / 'room/comparison.mp4').read_bytes() == b'updated video with more frames'
    assert (destination / 'local_notes.txt').read_text() == 'keep my notes'
    assert (destination / 'index.html').read_text() == 'gallery'
