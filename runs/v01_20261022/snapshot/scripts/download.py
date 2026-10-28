"""Explicit allowlist downloader; no snapshot_download and no inference APIs."""
from pathlib import Path
import yaml
from provenance import fetch
root=Path(__file__).resolve().parents[1]
cfg=yaml.safe_load((root/'configs/v0.yaml').read_text())
items=[]
for name in ['config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','README.md','model.safetensors']:
    items.append((f"https://huggingface.co/{cfg['donor']['repo']}/resolve/{cfg['donor']['revision']}/{name}",root/'artifacts/donor'/name))
items.extend([(f"https://huggingface.co/datasets/{cfg['data']['repo']}/resolve/{cfg['data']['revision']}/{cfg['data']['source_file']}",root/'data/source_train.parquet'),
              (f"https://huggingface.co/datasets/{cfg['data']['repo']}/resolve/{cfg['data']['revision']}/README.md",root/'data/DATASET_CARD.md')])
for url,path in items:
    path.parent.mkdir(parents=True,exist_ok=True)
    fetch(url,path)
