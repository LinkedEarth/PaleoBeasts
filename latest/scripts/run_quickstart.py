"""Run every Python code block on the Quick Start page as one script.

Extracts each fenced ```python code block from
``docs/get-started/quickstart.qmd`` in document order and executes them
sequentially in a single shared namespace, exactly as a reader following
the page top-to-bottom would. This both smoke-tests that the snippets stay
runnable and regenerates the figures referenced by the page (each block's
``plt.savefig(...)`` writes into ``docs/get-started/figures/``).

Usage
-----
```
python scripts/run_quickstart.py
```
"""
# Agg backend must be set before any other matplotlib import.
import matplotlib
matplotlib.use('Agg')

import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
DOCS_DIR = SCRIPT_DIR.parent            # docs/
PROJECT_DIR = DOCS_DIR.parent           # project root — contains climatecritters/
PAGE = DOCS_DIR / 'get-started' / 'quickstart.qmd'

FENCE_RE = re.compile(r'```python\s*\n(.*?)```', re.DOTALL)

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def extract_blocks(qmd_text: str) -> list[str]:
    """Return the body of every fenced ```python block, in document order."""
    return FENCE_RE.findall(qmd_text)


def main() -> None:
    if not PAGE.exists():
        print(f'Page not found: {PAGE}')
        sys.exit(1)

    blocks = extract_blocks(PAGE.read_text(encoding='utf-8'))
    if not blocks:
        print('No python code blocks found.')
        sys.exit(0)

    print(f'Running {len(blocks)} code block(s) from {PAGE.relative_to(PROJECT_DIR)}...')

    # Figure paths in the page are relative to the page's own directory
    # (e.g. 'figures/quickstart_integrate.png'), so run from there.
    namespace = {'__name__': '__main__', '__builtins__': __builtins__}
    os.chdir(PAGE.parent)

    for i, block in enumerate(blocks, start=1):
        print(f'  [{i}/{len(blocks)}]')
        try:
            exec(compile(block, f'<quickstart.qmd:block {i}>', 'exec'), namespace)  # noqa: S102
        except Exception:
            print(f'Block {i} failed:\n{block}')
            raise
        finally:
            plt.close('all')

    print('Done. All blocks ran successfully.')


if __name__ == '__main__':
    main()
