"""Export text evidence and privacy-redacted review bundles, without experiments.

No tensor, corpus, private file content or environment is opened. The source tree
and original review bundles remain unchanged. The export is a separate Git tree.
"""
import hashlib
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / 'publication/github'
TEXT_SUFFIXES = {'.py', '.md', '.txt', '.json', '.jsonl', '.yaml', '.yml', '.csv', '.log', '.diff'}
BLOCKED = {'.venv', '.git', 'private', 'artifacts', 'data', '__pycache__'}
PROFILE = re.compile(r'(?i)([a-z]:[\\/]+Users[\\/]+)([^\\/\r\n"\'<>]+)')
GPU_UUID = re.compile(r'\bGPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\b')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def serialized(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')


def redact(data):
    text = data.decode('utf-8')
    text, profiles = PROFILE.subn(lambda m: m.group(1) + 'USER', text)
    text, devices = GPU_UUID.subn('GPU-REDACTED', text)
    return text.encode('utf-8'), {k: v for k, v in {'windows_user_profile': profiles, 'gpu_unique_uuid': devices}.items() if v}


def safe_name(name):
    p = PurePosixPath(name)
    assert not p.is_absolute() and '..' not in p.parts and not BLOCKED.intersection(p.parts), name
    assert p.suffix.lower() in TEXT_SUFFIXES or p.name in {'.gitignore', '.gitattributes'}, name


def bundle(original, target):
    entries = {}
    records = {}
    with zipfile.ZipFile(original) as source:
        assert source.testzip() is None
        for name in source.namelist():
            safe_name(name)
            data = source.read(name)
            public, changes = redact(data)
            new_name = name.replace('/BUNDLE_MANIFEST.json', '/ORIGINAL_BUNDLE_MANIFEST.json')
            assert new_name not in entries
            entries[new_name] = public
            records[new_name] = {'bytes': len(public), 'sha256': sha(public), 'original_path': name,
                                 'original_bytes': len(data), 'original_sha256': sha(data), 'redactions': changes}
    metadata = {'kind': 'privacy-redacted derivative, not original execution bytes',
                'original_bundle_filename': original.name, 'original_bundle_bytes': original.stat().st_size,
                'original_bundle_sha256': sha(original.read_bytes()), 'files': records,
                'manifest_excludes_itself': True, 'original_manifest_is_historical_only': True}
    entries['PUBLIC_BUNDLE_MANIFEST.json'] = serialized(metadata)
    with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
        for name, data in sorted(entries.items()):
            output.writestr(name, data)
    return metadata


def main():
    assert not DEST.exists(), 'Export already exists; preserve it instead of overwriting'
    DEST.mkdir(parents=True)
    source_paths = set()
    for folder in ('scripts', 'src', 'tests', 'configs', 'reports'):
        for path in (ROOT / folder).rglob('*'):
            if path.is_file() and not BLOCKED.intersection(path.relative_to(ROOT).parts) and path.suffix.lower() in TEXT_SUFFIXES:
                source_paths.add(path)
    for folder in ('v0_20261021', 'v01_20261022', 'v02_20261022_193903'):
        for path in (ROOT / 'runs' / folder).rglob('*'):
            if path.is_file() and not BLOCKED.intersection(path.relative_to(ROOT).parts) and path.suffix.lower() in TEXT_SUFFIXES:
                source_paths.add(path)
    for path in (ROOT / 'runs').iterdir():
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES and not path.name.startswith(('install', 'download')):
            source_paths.add(path)
    for name in ('README.md', 'README_V0_HISTORY.md', 'RESEARCH_STATUS.md', 'PUBLICATION.md', '.gitignore', '.gitattributes',
                 'requirements.txt', 'CODEX_V01_VocabPriorBench.txt', 'CODEX_V02_VocabPriorBench.txt'):
        source_paths.add(ROOT / name)
    files = {}
    for path in sorted(source_paths):
        name = path.relative_to(ROOT).as_posix()
        safe_name(name)
        original = path.read_bytes()
        public, changes = redact(original)
        target = DEST / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(public)
        files[name] = {'bytes': len(public), 'sha256': sha(public), 'original_bytes': len(original),
                       'original_sha256': sha(original), 'redactions': changes}
    original_bundles = {}
    for phase in ('V01', 'V02'):
        original = ROOT / (phase + '_REVIEW_BUNDLE.zip')
        name = phase + '_PUBLIC_REVIEW_BUNDLE.zip'
        target = DEST / name
        meta = bundle(original, target)
        original_bundles[original.name] = {k: meta[k] for k in ('original_bundle_bytes', 'original_bundle_sha256')}
        files[name] = {'bytes': target.stat().st_size, 'sha256': sha(target.read_bytes()),
                       'source_bundle': original.name, 'privacy_redacted_derivative': True}
        checksum_name = name + '.sha256'
        checksum = (files[name]['sha256'] + '  ' + name + '\n').encode('ascii')
        (DEST / checksum_name).write_bytes(checksum)
        files[checksum_name] = {'bytes': len(checksum), 'sha256': sha(checksum), 'generated': True}
    manifest = {'schema_version': 1, 'publication_date': '2026-10-28',
                'generated_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                'github_repository': 'https://github.com/tianyi0703/VocabPriorBench',
                'research_status': 'route paused; freezing/sign paper proposal closed',
                'publication_is_not_an_experiment_or_preregistration': True,
                'original_files_and_bundles_unchanged': True, 'original_bundles': original_bundles,
                'redactions': ['Windows user-profile names', 'unique GPU UUIDs'],
                'excluded': sorted(BLOCKED | {'all tensors/weights/corpora', 'credentials', 'installation/download logs'}),
                'manifest_excludes_itself': True, 'files': files}
    (DEST / 'PUBLICATION_MANIFEST.json').write_bytes(serialized(manifest))
    print(json.dumps({'export': str(DEST), 'files_excluding_manifest': len(files),
                      'redacted_text_files': sum(bool(x.get('redactions')) for x in files.values()),
                      'bytes_excluding_manifest': sum(x['bytes'] for x in files.values())}, indent=2))


if __name__ == '__main__':
    main()
