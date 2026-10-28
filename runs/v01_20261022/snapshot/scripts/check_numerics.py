"""Run CPU tests and keep an encoding-stable raw transcript."""
import subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
with open(root/'runs/numerical_checks.log','wb') as output:
    result=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=root,stdout=output,stderr=subprocess.STDOUT)
print((root/'runs/numerical_checks.log').read_text(encoding='utf-8',errors='replace'))
raise SystemExit(result.returncode)
