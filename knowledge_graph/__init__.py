"""
Compatibility package for direct repo-root ``python -m`` commands.
"""

from pathlib import Path
from pkgutil import extend_path


__path__ = extend_path(__path__, __name__)

SRC_PACKAGE_DIR = Path(__file__).resolve().parents[1] / "src" / "knowledge_graph"
if SRC_PACKAGE_DIR.is_dir():
    __path__.append(str(SRC_PACKAGE_DIR))
