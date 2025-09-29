# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

try:
    import sphinx_toolbox.utils as _stu
except Exception:
    _stu = None

if _stu is not None and hasattr(_stu, "flag"):
    def _safe_flag(argument):
        # Accetta bool/None come "flag presente"
        if isinstance(argument, (bool, type(None))):
            return True
        # Se qualcuno ha scritto per sbaglio un valore testuale per un flag, emula il comportamento moderno:
        if isinstance(argument, str) and argument.strip():
            raise ValueError(f"flag options must not have an argument (got {argument!r})")
        return True
    _stu.flag = _safe_flag

HERE = Path(__file__).resolve()

def find_repo_root(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / ".env").exists() or (p / "pyproject.toml").exists():
            return p
    for p in [start] + list(start.parents):
        if (p / "src").exists():
            return p
    return start.parent.parent

REPO_ROOT = find_repo_root(HERE)
sys.path.insert(0, str(REPO_ROOT))

# Carica il .env reale
if load_dotenv:
    load_dotenv(REPO_ROOT / ".env")

# Flag per codice “doc-friendly”
os.environ.setdefault("SPHINX_BUILD", "1")

project = 'Cardiology GenAI - Data ETL'
# copyright = '2025, gaia'
# author = 'gaia'
release = '0.1.0'

import inspect

def _skip_imported_members(app, what, name, obj, skip, options):
    current_mod = app.env.temp_data.get('autodoc:module')
    if not current_mod:
        return None
    obj_mod = getattr(obj, "__module__", None)
    if what == "module":
        if obj_mod and not obj_mod.startswith(current_mod):
            return True
        return None
    if what == "class":
        if obj_mod and not obj_mod.startswith(current_mod):
            return True
        return None
    if obj_mod and not obj_mod.startswith(current_mod):
        return True
    return None

def setup(app):
    app.connect("autodoc-skip-member", _skip_imported_members)


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
    "sphinx.ext.graphviz",
    "sphinx.ext.extlinks",

    # "sphinxcontrib.autodoc_pydantic",
    "sphinxcontrib.mermaid",
    "sphinx_toolbox.more_autodoc.autotypeddict",
]

mermaid_output_format = "raw"
mermaid_init_js = "mermaid.initialize({startOnLoad:true});"
autosummary_generate = True

autodoc_default_options = {
    "members": True,
    "imported-members": False,
    "undoc-members": False,
    "private-members": True,
    "inherited-members": False,
    "member-order": "bysource",
    "exclude-members": (
        "model_config,model_fields,model_computed_fields,model_extra,"
        "model_parametrized_name,model_rebuild,"
        "model_json_schema,model_validate,model_dump"
    )
}

nitpicky = True  # error when unresolved ref are detected
nitpick_ignore = [
    ("py:class", "optional"),
]

html_theme = 'sphinx_book_theme'  # "furo"  # TODO: change if needed

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
    "langchain": ("https://python.langchain.com/api_reference/", None),
    "transformers": ("https://huggingface.co/transformers/v2.11.0/", None),
    "langchain_huggingface": ("https://python.langchain.com/api_reference/huggingface",
        "https://python.langchain.com/api_reference/objects.inv",),
    "langchain_community": ("https://python.langchain.com/api_reference/community/",
        "https://python.langchain.com/api_reference/objects.inv",),
    "langchain_qdrant": ("https://python.langchain.com/api_reference/qdrant/",
        "https://python.langchain.com/api_reference/objects.inv",),
    "pymupdf4llm": ("https://pymupdf.readthedocs.io/en/latest/pymupdf4llm", "https://pymupdf.qubitpi.org/en/latest/"),
    "cardiology_gen_ai": ("https://cardiology-gen-ai.github.io/cardiology-gen-ai", None)
}

extlinks = {
    "langgraph": ("https://langchain-ai.github.io/langgraph/%s", "%s"),
    "langchain_core": ("https://python.langchain.com/api_reference/core/%s", "%s"),
    "pymupdf": ("https://pymupdf.readthedocs.io/en/latest/%s", "%s"),
    "pymupdf4llm": ("https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/%s", "%s"),
    "langchain": ("https://python.langchain.com/api_reference/%s", "%s"),
    "pydantic": ("https://docs.pydantic.dev/latest/api/%s", "%s"),
    "faiss": ("https://faiss.ai/cpp_api/%s", "%s"),
    "qdrant": ("https://python-client.qdrant.tech/%s", "%s")
}

autodoc_typehints = "none"
autodoc_mock_imports = []

templates_path = ['_templates']
exclude_patterns = ['.DS_Store']

STATIC_DIR = Path(__file__).parent / "_static"
html_static_path = ["_static"] if STATIC_DIR.exists() else []
