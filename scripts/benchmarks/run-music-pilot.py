#!/usr/bin/env python3
"""Run the frozen music pilot using the benchmark family's generation driver.

A limited run retains the full expected cohort and labels unattempted outputs.
Configuration JSON requires name, checkpoint_identity, backend, build_options.
Checkpoint identity names the original checkpoint, shared by quantized variants.
"""
import argparse
import hashlib
import json
import math
import os
import signal
import re
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def normalize_score(score):
    if not isinstance(score, dict):
        return {'status': 'scorer-error', 'score': None, 'reason': 'malformed score object'}
    score = dict(score)
    value = score.get('score')
    if score.get('status') == 'ok':
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not -1 <= value <= 1:
            return {'status': 'scorer-error', 'score': None, 'reason': 'invalid cosine score'}
    else:
        score['score'] = None
    return score


def validate_manifest(manifest):
    seeds = manifest['seeds']
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('manifest seeds must be unique nonnegative integers')
    ids = set()
    for prompt in manifest['prompts']:
        identifier = prompt['id']
        if not re.fullmatch(r'[a-zA-Z0-9_-]+', identifier) or identifier in ids:
            raise ValueError('manifest prompt IDs must be unique safe path components')
        ids.add(identifier)
        if prompt['subset'] not in ('primary', 'stress') or not prompt['caption'].strip():
            raise ValueError('manifest prompt requires a subset and nonempty caption')
    if not ids:
        raise ValueError('manifest has no prompts')
    if manifest['bootstrap']['replicates'] < 100:
        raise ValueError('bootstrap requires at least 100 replicates')


def extract_score(path):
    if not path.exists():
        return {'status': 'scorer-error', 'score': None, 'reason': 'scorer did not write JSON'}
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError) as error:
        return {'status': 'scorer-error', 'score': None, 'reason': str(error)}


def run_bounded(command, *, timeout, **kwargs):
    """Kill the whole driver process group on timeout, including generators."""
    with subprocess.Popen(command, start_new_session=True, **kwargs) as process:
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise


def score_control(wav, caption, destination, args):
    command = [os.environ.get('MUSIC_ALIGNMENT_PYTHON', sys.executable),
               str(HERE / 'compute-music-alignment.py'), '--wav', str(wav),
               '--caption', caption, '--model-dir', str(args.scorer_model_dir),
               '--model-manifest', str(args.scorer_manifest), '--json-out', str(destination)]
    try:
        with destination.with_suffix('.log').open('w') as log:
            run_bounded(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.scorer_timeout)
        return normalize_score(extract_score(destination))
    except subprocess.TimeoutExpired:
        return {'status': 'timeout', 'score': None, 'reason': 'control scorer timeout'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=['acestep', 'minimax'], required=True)
    parser.add_argument('--configuration', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=HERE / 'fixtures/music-pilot-v1.json')
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--models-root', type=Path, default=Path('bench-models'))
    parser.add_argument('--model-dir', type=Path, help='Provisioned generation models; bypass registry')
    parser.add_argument('--scorer-model-dir', type=Path, required=True)
    parser.add_argument('--scorer-manifest', type=Path, default=HERE / 'music-alignment-model.json')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--limit', type=int, help='Explicit partial smoke cohort, including all missing IDs in report')
    parser.add_argument('--generation-timeout', type=float, default=3600)
    parser.add_argument('--scorer-timeout', type=float, default=300)
    parser.add_argument('--controls', action='store_true', help='Same-WAV fresh-process and different-category caption controls; one per category')
    parser.add_argument('--repeat-generation', action='store_true', help='Repeat first successful prompt/seed generation once')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    manifest = json.loads(args.manifest.read_text())
    try:
        validate_manifest(manifest)
    except (ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    config = json.loads(args.configuration.read_text())
    for key in ('name', 'checkpoint_identity', 'backend', 'build_options'):
        if key not in config or not config[key]:
            parser.error(f'configuration requires {key}')
    device = str(config['backend']).lower()
    if device not in ('cpu', 'gpu'):
        parser.error('configuration backend must be cpu or gpu; requested device is applied through MUSIC_DEVICE')
    if args.out_dir.exists():
        parser.error('--out-dir must be new to prevent overwriting retained evidence')
    args.out_dir.mkdir(parents=True)
    output = args.out_dir.resolve()
    settings = manifest['generation'][args.family]
    config['generation_settings'] = settings
    config['source_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip()
    config['source_diff_sha256'] = hashlib.sha256(subprocess.check_output(['git', 'diff', 'HEAD'], cwd=HERE)).hexdigest()
    config['source_dirty'] = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=HERE, text=True).strip())
    config['source_pinned'] = not config['source_dirty']
    config['benchmark_source_hashes'] = {str(p.relative_to(HERE)): digest(p) for p in sorted(HERE.rglob('*')) if p.is_file() and p.suffix in ('.py', '.sh', '.json', '.txt')}
    pilot = {'schema_version': 1, 'manifest': manifest, 'manifest_sha256': digest(args.manifest),
             'family': args.family, 'configuration': config, 'records': [], 'controls': [],
             'listening_review': 'pending human observations',
             'validity_controls': 'see model-free scorer tests; empirical corruption controls not run'}
    write(output / 'manifest.json', manifest)
    write(output / 'configuration.json', config)
    env = os.environ.copy()
    env.update(MUSIC_ALIGNMENT='1', MUSIC_DEVICE=device, MUSIC_ALIGNMENT_MODEL=str(args.scorer_model_dir.resolve()),
               MUSIC_ALIGNMENT_MANIFEST=str(args.scorer_manifest.resolve()),
               MUSIC_ALIGNMENT_TIMEOUT=str(args.scorer_timeout))
    if args.model_dir:
        env['MUSIC_ALIGNMENT_MODEL_DIR'] = str(args.model_dir.resolve())
    categories = set()
    repeated = False
    attempted = 0
    for prompt in manifest['prompts']:
        for seed in manifest['seeds']:
            row = {'prompt_id': prompt['id'], 'seed': seed, 'status': 'not-run', 'score': None,
                   'reason': 'outside explicitly limited cohort' if args.limit is not None else 'not attempted'}
            pilot['records'].append(row)
            if args.limit is not None and attempted >= args.limit:
                continue
            attempted += 1
            run_dir = output / f"{prompt['id']}-{seed}"
            run_dir.mkdir()
            result_path = run_dir / 'result.json'
            env.update(MUSIC_CAPTION=prompt['caption'], MUSIC_LYRICS=prompt['lyrics'], MUSIC_SEED=str(seed),
                       MUSIC_DURATION=str(settings.get('duration_seconds') or 20), MUSIC_MAX_FRAMES=str(settings.get('max_frames') or 300))
            command = ['bash', str(HERE / 'run-family.sh'), '--family', args.family, '--runs', '1', '--warmup', '0',
                       '--build-dir', str(args.build_dir.resolve()), '--models-root', str(args.models_root.resolve()), '--out', str(result_path)]
            started = time.monotonic()
            try:
                with (run_dir / 'driver.log').open('w') as log:
                    run_bounded(command, cwd=HERE.parent.parent, env=env, stdout=log, stderr=subprocess.STDOUT,
                                timeout=args.generation_timeout)
                result = extract_score(result_path) if result_path.exists() else {
                    'status': 'run-failed', 'notes': 'generation driver did not write result.json; see driver.log'}
                scored_runs = result.get('music_alignment', {}).get('runs', [])
                score = normalize_score(scored_runs[0] if scored_runs else result.get('music_alignment', {}))
                row.update(status=score.get('status', result.get('status', 'run-failed')),
                           score=score.get('score'), reason=score.get('reason') or result.get('notes'),
                           scorer_provenance=({'policy_version': score.get('policy_version'), **score['provenance']} if score.get('provenance') else None), result=str(result_path.relative_to(output)))
                row['performance_result'] = result
                artifact_dir = score.get('artifact_dir')
                generation_file = Path(artifact_dir) / 'generation.json' if artifact_dir else None
                row['generation'] = json.loads(generation_file.read_text()) if generation_file and generation_file.is_file() else None
                row['score_artifact'] = score
            except subprocess.TimeoutExpired:
                row.update(status='timeout', reason='generation driver timeout')
            row['elapsed_seconds'] = time.monotonic() - started
            wav_paths = sorted(run_dir.glob('result.music.*/run-*/audio.wav'))
            if row['status'] == 'ok' and wav_paths:
                wav = wav_paths[0]
                if args.controls and prompt['category'] not in categories and prompt['subset'] == 'primary':
                    categories.add(prompt['category'])
                    mismatch = next(p for p in manifest['prompts'] if p['subset'] == 'primary' and p['category'] != prompt['category'])
                    pilot['controls'].append({'prompt_id': prompt['id'], 'seed': seed, 'category': prompt['category'], 'matched': row['score_artifact'], 'wav_sha256': digest(wav),
                        'repeat': score_control(wav, prompt['caption'], run_dir / 'repeat-score.json', args),
                        'mismatch_prompt_id': mismatch['id'],
                        'mismatch': score_control(wav, mismatch['caption'], run_dir / 'mismatch-score.json', args)})
                if args.repeat_generation and not repeated:
                    repeated = True
                    repeat_path = run_dir / 'repeat-result.json'
                    repeat_command = command[:-1] + [str(repeat_path)]
                    try:
                        with (run_dir / 'repeat-driver.log').open('w') as log:
                            run_bounded(repeat_command, cwd=HERE.parent.parent, env=env, stdout=log,
                                        stderr=subprocess.STDOUT, timeout=args.generation_timeout)
                        pilot['controls'].append({'kind': 'repeat-generation', 'prompt_id': prompt['id'],
                                                  'seed': seed, 'result': extract_score(repeat_path)})
                    except subprocess.TimeoutExpired:
                        pilot['controls'].append({'kind': 'repeat-generation', 'status': 'timeout', 'prompt_id': prompt['id'], 'seed': seed})
            write(output / 'pilot.json', pilot)
    write(output / 'pilot.json', pilot)
    print(output / 'pilot.json')


if __name__ == '__main__':
    main()
