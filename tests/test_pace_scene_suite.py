"""PACE runner preserves arguments, compute boundaries and process failures."""

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest


RUNNER = Path(__file__).resolve().parents[1] / 'scripts/pace/run_scene_suite.sh'


@pytest.fixture
def runner_project(tmp_path):
    project = tmp_path / 'project with spaces'
    (project / 'scripts').mkdir(parents=True)
    (project / 'pyproject.toml').write_text('[project]\nname="fixture"\n')
    (project / 'scripts/run_scene_suite.py').write_text('# Fixture entry point\n')
    record = tmp_path / 'invocation.json'
    python = project / '.venv/bin/python'
    python.parent.mkdir(parents=True)
    implementation = tmp_path / 'fake_python.py'
    implementation.write_text('''
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
output = Path(args[args.index('--output') + 1])
keys = ('MUJOCO_GL', 'MPLBACKEND', 'PYTHONDONTWRITEBYTECODE', 'PYTHONUNBUFFERED',
        'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
        'NUMEXPR_NUM_THREADS', 'MPLCONFIGDIR', 'XDG_CACHE_HOME')
Path(os.environ['TEST_INVOCATION']).write_text(json.dumps({
    'args': args, 'cwd': str(Path.cwd()), 'output_preexists': output.exists(),
    'environment': {key: os.environ.get(key) for key in keys}}))
print('fake experiment output', flush=True)
code = int(os.environ.get('TEST_EXIT_CODE', '0'))
if not code:
    output.mkdir(parents=True, exist_ok='--replace-scenes' in args)
    (output / 'index.html').write_text('fixture gallery')
sys.exit(code)
''')
    python.write_text('#!/usr/bin/env bash\nexec ' + shlex.quote(sys.executable) + ' ' +
                      shlex.quote(str(implementation)) + ' "$@"\n')
    python.chmod(0o755)
    environment = os.environ.copy()
    for name in ('SLURM_JOB_ID', 'SLURM_JOBID', 'SLURM_STEP_ID', 'SLURM_CPUS_PER_TASK',
                 'WTUMI_PYTHON', 'WTUMI_SUITE_OUTPUT', 'XDG_CACHE_HOME'):
        environment.pop(name, None)
    environment.update(WTUMI_WORKDIR=str(project), TEST_INVOCATION=str(record))
    return project, record, environment


def _run(environment, *args):
    return subprocess.run(['bash', str(RUNNER), *args], env=environment,
                          capture_output=True, text=True)


def _allocated(environment):
    return {**environment, 'SLURM_JOB_ID': '12345', 'SLURM_STEP_ID': '0',
            'SLURM_CPUS_PER_TASK': '6'}


def _existing_suite(project):
    result = project / 'existing suite'
    result.mkdir()
    (result / 'suite.json').write_text('{"scene_order": ["room", "doorway", "g1_stairs"]}\n')
    (result / 'g1_stairs').mkdir()
    (result / 'g1_stairs/comparison.mp4').write_bytes(b'preserve existing stairs video')
    return result


def test_dry_run_needs_no_allocation_or_python_and_creates_nothing(runner_project):
    project, record, environment = runner_project
    (project / '.venv/bin/python').unlink()
    result = _run(environment, '--dry-run', '--output', 'outputs/my 30fps run')
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout.splitlines()[-1])
    assert command[-5:] == [str(project / 'scripts/run_scene_suite.py'), '--preset',
                           'pace30', '--output', str(project / 'outputs/my 30fps run')]
    assert 'MUJOCO_GL=egl' in command and 'OMP_NUM_THREADS=8' in command
    assert not record.exists()
    assert not (project / 'outputs').exists()


@pytest.mark.parametrize('allocation', [{}, {'SLURM_JOB_ID': '12345'}, {'SLURM_STEP_ID': '0'}])
def test_login_shell_and_salloc_without_a_compute_step_are_rejected(runner_project, allocation):
    project, record, environment = runner_project
    result = _run({**environment, **allocation})
    assert result.returncode == 2
    assert 'compute-node shell' in result.stderr
    assert not record.exists()
    assert not (project / 'outputs').exists()


def test_allocated_run_preserves_literal_arguments_and_uses_separate_logs(runner_project):
    project, record, environment = runner_project
    result_name = 'outputs/run with spaces $(touch never-execute)'
    result = _run(_allocated(environment), '--output', result_name)
    assert result.returncode == 0, result.stderr
    invocation = json.loads(record.read_text())
    destination = project / result_name
    assert invocation['args'] == [str(project / 'scripts/run_scene_suite.py'),
                                  '--preset', 'pace30', '--output', str(destination)]
    assert invocation['cwd'] == str(project)
    assert invocation['output_preexists'] is False
    assert not (project / 'never-execute').exists()
    assert (destination / 'index.html').is_file()
    values = invocation['environment']
    assert values['MUJOCO_GL'] == 'egl' and values['MPLBACKEND'] == 'Agg'
    assert values['OMP_NUM_THREADS'] == '6'
    assert values['PYTHONDONTWRITEBYTECODE'] == values['PYTHONUNBUFFERED'] == '1'
    assert all(values[key] == '1' for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'))
    cache = Path(values['MPLCONFIGDIR'])
    assert cache.is_relative_to(project / 'outputs/logs')
    assert not cache.is_relative_to(destination)
    logs = list((project / 'outputs/logs').glob('*/run.log'))
    assert len(logs) == 1
    assert 'fake experiment output' in logs[0].read_text()
    assert f'Completed result: {destination}' in result.stdout
    assert f'Gallery: {destination}/index.html' in result.stdout


def test_python_failure_survives_tee_and_does_not_report_completion(runner_project):
    project, record, environment = runner_project
    result = _run({**_allocated(environment), 'TEST_EXIT_CODE': '7'}, '--output', 'outputs/failing')
    assert result.returncode == 7
    assert record.is_file()
    assert 'Completed result:' not in result.stdout
    log, = (project / 'outputs/logs').glob('*/run.log')
    assert 'fake experiment output' in log.read_text()


@pytest.mark.parametrize('kind', ['file', 'directory', 'dangling_symlink'])
def test_existing_result_is_preserved_before_python_or_log_creation(runner_project, kind):
    project, record, environment = runner_project
    destination = project / 'existing'
    if kind == 'file':
        destination.write_text('keep')
    elif kind == 'directory':
        destination.mkdir()
        (destination / 'keep.txt').write_text('keep')
    else:
        destination.symlink_to(project / 'missing')
    result = _run(_allocated(environment), '--output', str(destination))
    assert result.returncode == 2
    assert 'already exists' in result.stderr
    assert not record.exists() and not (project / 'outputs').exists()
    if kind == 'file':
        assert destination.read_text() == 'keep'
    elif kind == 'directory':
        assert (destination / 'keep.txt').read_text() == 'keep'
    else:
        assert destination.is_symlink()


def test_missing_python_reports_actionable_error_without_creating_logs(runner_project):
    project, record, environment = runner_project
    result = _run({**_allocated(environment), 'WTUMI_PYTHON': str(project / 'missing python')})
    assert result.returncode == 2 and 'WTUMI_PYTHON' in result.stderr
    assert not record.exists() and not (project / 'outputs').exists()


def test_output_and_python_environment_overrides_and_batch_step(runner_project):
    project, record, environment = runner_project
    alternative = project / 'alternate python'
    (project / '.venv/bin/python').rename(alternative)
    result = _run({**_allocated(environment), 'SLURM_STEP_ID': 'batch',
                   'WTUMI_PYTHON': str(alternative), 'WTUMI_SUITE_OUTPUT': 'outputs/from environment'})
    assert result.returncode == 0, result.stderr
    assert json.loads(record.read_text())['args'][-1] == str(project / 'outputs/from environment')


@pytest.mark.parametrize('args', [('--output',), ('--unknown',), ('--output', '--dry-run')])
def test_invalid_arguments_do_not_create_outputs(runner_project, args):
    project, record, environment = runner_project
    result = _run(environment, *args)
    assert result.returncode == 2
    assert not record.exists() and not (project / 'outputs').exists()


def test_invalid_slurm_cpu_count_fails_without_execution(runner_project):
    project, record, environment = runner_project
    result = _run({**_allocated(environment), 'SLURM_CPUS_PER_TASK': '0'})
    assert result.returncode == 2 and 'positive integer' in result.stderr
    assert not record.exists() and not (project / 'outputs').exists()


def test_default_output_is_printed_exactly_for_download(runner_project):
    project, record, environment = runner_project
    result = _run(_allocated(environment))
    assert result.returncode == 0, result.stderr
    destination = Path(json.loads(record.read_text())['args'][-1])
    assert destination.name.startswith('g1_vision_30fps_pace_')
    assert f'Result directory: {destination}' in result.stdout
    assert f'Result folder name: {destination.name}' in result.stdout


def test_replacement_forwards_selected_scenes_and_preserves_existing_stairs(runner_project):
    project, record, environment = runner_project
    destination = _existing_suite(project)
    result = _run(_allocated(environment), '--replace-scenes', 'room', 'doorway',
                  '--output', destination.name)
    assert result.returncode == 0, result.stderr
    invocation = json.loads(record.read_text())
    assert invocation['args'] == [str(project / 'scripts/run_scene_suite.py'),
                                  '--preset', 'pace30', '--output', str(destination),
                                  '--replace-scenes', 'room', 'doorway']
    assert invocation['output_preexists'] is True
    assert (destination / 'g1_stairs/comparison.mp4').read_bytes() == b'preserve existing stairs video'
    assert (destination / 'suite.json').is_file()
    logs, = (project / 'outputs/logs').glob('*/run.log')
    assert 'fake experiment output' in logs.read_text()


def test_replacement_dry_run_uses_environment_output_without_python_or_new_files(runner_project):
    project, record, environment = runner_project
    destination = _existing_suite(project)
    before = {path.relative_to(project) for path in project.rglob('*')}
    (project / '.venv/bin/python').unlink()
    before.remove(Path('.venv/bin/python'))
    result = _run({**environment, 'WTUMI_SUITE_OUTPUT': str(destination)},
                  '--replace-scenes', 'room', 'doorway', '--dry-run')
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout.splitlines()[-1])
    assert command[-8:] == [str(project / 'scripts/run_scene_suite.py'),
                            '--preset', 'pace30', '--output', str(destination),
                            '--replace-scenes', 'room', 'doorway']
    assert not record.exists()
    assert {path.relative_to(project) for path in project.rglob('*')} == before


@pytest.mark.parametrize('args', [
    ('--replace-scenes',),
    ('--replace-scenes', '--dry-run'),
    ('--replace-scenes', 'room', 'room'),
    ('--replace-scenes', 'room', '--replace-scenes', 'room'),
    ('--replace-scenes', 'room', '--replace-scenes', '--dry-run'),
    ('--replace-scenes', 'room', 'unknown'),
    ('--replace-scenes', '../room'),
])
def test_invalid_replacement_selections_do_not_execute_or_create_logs(runner_project, args):
    project, record, environment = runner_project
    destination = _existing_suite(project)
    result = _run(_allocated(environment), '--output', str(destination), *args)
    assert result.returncode == 2
    assert not record.exists() and not (project / 'outputs').exists()
    assert (destination / 'g1_stairs/comparison.mp4').read_bytes() == b'preserve existing stairs video'


@pytest.mark.parametrize('dry_run', [False, True])
def test_replacement_requires_an_explicit_output(runner_project, dry_run):
    project, record, environment = runner_project
    args = ['--replace-scenes', 'room', 'doorway']
    if dry_run:
        args.append('--dry-run')
    result = _run(_allocated(environment), *args)
    assert result.returncode == 2
    assert '--output or WTUMI_SUITE_OUTPUT' in result.stderr
    assert not record.exists() and not (project / 'outputs').exists()


@pytest.mark.parametrize('kind', ['missing', 'file', 'unrecognized_directory',
                                 'directory_symlink', 'trailing_slash_symlink',
                                 'metadata_symlink', 'metadata_directory'])
def test_replacement_requires_a_recognized_non_symlink_suite(runner_project, kind):
    project, record, environment = runner_project
    destination = project / 'destination'
    if kind == 'file':
        destination.write_text('keep')
    elif kind == 'unrecognized_directory':
        destination.mkdir()
    elif kind in ('directory_symlink', 'trailing_slash_symlink'):
        destination.symlink_to(_existing_suite(project), target_is_directory=True)
    elif kind in ('metadata_symlink', 'metadata_directory'):
        destination.mkdir()
        if kind == 'metadata_symlink':
            metadata = project / 'metadata.json'
            metadata.write_text('{}')
            (destination / 'suite.json').symlink_to(metadata)
        else:
            (destination / 'suite.json').mkdir()
    output_arg = str(destination) + ('/' if kind == 'trailing_slash_symlink' else '')
    result = _run(_allocated(environment), '--output', output_arg,
                  '--replace-scenes', 'room')
    assert result.returncode == 2
    assert 'existing suite directory' in result.stderr
    assert not record.exists() and not (project / 'outputs').exists()


def test_recognized_suite_still_rejects_overwrite_without_explicit_replacement(runner_project):
    project, record, environment = runner_project
    destination = _existing_suite(project)
    result = _run(_allocated(environment), '--output', str(destination))
    assert result.returncode == 2 and 'already exists' in result.stderr
    assert not record.exists() and not (project / 'outputs').exists()


def test_selected_scene_replacement_keeps_python_failure_and_stairs(runner_project):
    project, record, environment = runner_project
    destination = _existing_suite(project)
    result = _run({**_allocated(environment), 'TEST_EXIT_CODE': '7'},
                  '--output', str(destination), '--replace-scenes', 'room', 'doorway')
    assert result.returncode == 7
    assert record.exists() and 'Completed result:' not in result.stdout
    assert (destination / 'g1_stairs/comparison.mp4').read_bytes() == b'preserve existing stairs video'


def test_replacement_accepts_a_single_valid_scene(runner_project):
    project, record, environment = runner_project
    destination = _existing_suite(project)
    result = _run(_allocated(environment), '--output', str(destination),
                  '--replace-scenes', 'g1_stairs')
    assert result.returncode == 0, result.stderr
    assert json.loads(record.read_text())['args'][-2:] == ['--replace-scenes', 'g1_stairs']
