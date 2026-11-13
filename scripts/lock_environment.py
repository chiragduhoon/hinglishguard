"""Record installed application dependency closure; excludes unrelated host packages."""
from importlib.metadata import distribution
from pathlib import Path
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
import platform

queue=['pandas','numpy','scikit-learn','scipy','joblib','pydantic','httpx','streamlit','pytest']
visited={}
while queue:
    name=canonicalize_name(queue.pop())
    if name in visited: continue
    dist=distribution(name); visited[name]=dist.version
    for raw in dist.requires or []:
        req=Requirement(raw)
        if req.marker and not req.marker.evaluate({'extra':''}): continue
        queue.append(req.name)
header=f'# Application dependencies tested on {platform.system()}, Python {platform.python_version()}.\n# GPU training extras are installed separately from pyproject.toml on Chirag.\n'
target=Path(__file__).resolve().parents[1]/'requirements.lock'
target.write_text(header+'\n'.join(f'{name}=={version}' for name,version in sorted(visited.items()))+'\n',encoding='utf-8')
print(f'Locked {len(visited)} application/test distributions')
