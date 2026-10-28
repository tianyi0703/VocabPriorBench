"""Verify cached bytes against immutable Hub revision metadata."""
import hashlib, json, re, urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
_metadata={}

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def fetch(url,path):
    match=re.fullmatch(r'https://huggingface.co/(datasets/)?([^/]+/[^/]+)/resolve/([0-9a-f]{40})/(.+)',url)
    if not match:raise ValueError('Only immutable pinned Hugging Face downloads are supported')
    dataset,repo,revision,name=match.groups();kind='datasets' if dataset else 'models'
    key=(kind,repo,revision)
    if key not in _metadata:
        api=f'https://huggingface.co/api/{kind}/{repo}/revision/{revision}?blobs=true'
        metadata=json.load(urllib.request.urlopen(api,timeout=60))
        assert metadata['sha']==revision
        _metadata[key]={entry['rfilename']:entry for entry in metadata['siblings']}
    expected=_metadata[key][name];path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        print(json.dumps({'event':'download_start','url':url,'path':str(path)}),flush=True)
        with urllib.request.urlopen(url,timeout=180) as response,open(str(path)+'.partial','wb') as stream:
            while chunk:=response.read(1024*1024):stream.write(chunk)
        Path(str(path)+'.partial').replace(path)
    actual_sha=sha(path)
    assert path.stat().st_size==expected['size'],f'Cached file size mismatch: {path}'
    if 'lfs' in expected:
        assert actual_sha==expected['lfs']['sha256'],f'Cached file SHA256 mismatch: {path}'
    else:
        content=path.read_bytes()
        blob=hashlib.sha1(f'blob {len(content)}\0'.encode()+content).hexdigest()
        assert blob==expected['blobId'],f'Cached Git blob mismatch: {path}'
    result={'url':url,'path':str(path.relative_to(ROOT)),'bytes':path.stat().st_size,
            'sha256':actual_sha,'remote_metadata':expected,'revision_metadata_verified':True}
    print(json.dumps({'event':'verified_file',**result}),flush=True)
    return result
