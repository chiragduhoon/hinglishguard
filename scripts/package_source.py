from pathlib import Path
import tarfile

root=Path(__file__).resolve().parents[1]
# Never ship credentials or local bot settings, even if they exist in config/.
SECRETS={'config/discord.token','config/discord.local.json','.env'}
target=root/'artifacts/deployment.tar.gz'
with tarfile.open(target,'w:gz') as archive:
    for path in root.rglob('*'):
        parts=path.relative_to(root).parts
        if any(p in ('.venv','__pycache__','.pytest_cache','checkpoints','runtime') or p.startswith('.test-') or p=='.git' or p.endswith('.egg-info') for p in parts): continue
        if path.relative_to(root).as_posix() in SECRETS: continue
        if path==target or not path.is_file() or path.suffix in ('.sqlite','.webm','.png','.pptx','.safetensors','.zst'): continue
        archive.add(path,arcname=Path('HinglishGuard')/path.relative_to(root))
print(target)
