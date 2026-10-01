from pathlib import Path
import sys, platform
mods=['numpy','pandas','PIL','yaml','sklearn','scipy','matplotlib','torch','streamlit','plotly']
print('Python:',sys.version); print('Platform:',platform.platform())
for m in mods:
    try:
        mod=__import__(m); print(f'{m}: OK',getattr(mod,'__version__',''))
    except Exception as e: print(f'{m}: MISSING ({e})')
