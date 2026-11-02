"""Name-keyed registries for routers and serving backends, and loading of plugin files that add to them."""
from __future__ import annotations

import difflib
import hashlib
import importlib.util
import os
import sys

ROUTERS: dict = {}   # name -> Router subclass
SERVING: dict = {}   # name -> Serving subclass


def _key(name: str) -> str:
    """Router and backend names use dashes; single_turn finds single-turn."""
    return name.replace("_", "-")


def register_router(cls=None, *, name: str | None = None):
    """Class decorator: make a Router subclass available by its name (config `router:`, --router)."""
    def wrap(c):
        n = _key(name or c.name or "")
        if not n:
            raise ValueError(f"router class {c.__qualname__} needs a name")
        old = ROUTERS.get(n)
        if old is not None and old is not c and old.__module__.startswith("multiturnserve."):
            raise ValueError(f"router {n!r} is built in ({old.__module__}.{old.__qualname__}); pick another name")
        c.name = n
        ROUTERS[n] = c
        return c
    return wrap(cls) if cls is not None else wrap


def register_serving(cls=None, *, name: str | None = None):
    """Class decorator: make a Serving subclass available by its name (config `serving:`)."""
    def wrap(c):
        n = _key(name or c.name or "")
        if not n:
            raise ValueError(f"serving class {c.__qualname__} needs a name")
        c.name = n
        SERVING[n] = c
        return c
    return wrap(cls) if cls is not None else wrap


def load_plugin(path: str):
    """Import a .py file that registers routers or backends (once per version of the file)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"plugin file {path} not found")
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()[:12]
    stem = "".join(ch if ch.isalnum() else "_" for ch in os.path.splitext(os.path.basename(path))[0])
    mod_name = f"mts_plugin_{stem}_{digest}"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    folder = os.path.dirname(os.path.abspath(path))
    if folder not in sys.path:
        sys.path.insert(0, folder)                 # so the plugin can import its neighbours (e.g. my_lab.py)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _builtins() -> None:
    from . import routers, serving  # noqa: F401  (registers the built-ins)


def get_router(spec: str):
    """A Router class by name, or the router a plugin file registers (my_router.py)."""
    _builtins()
    if spec.endswith(".py"):
        mod = load_plugin(spec)
        found = [c for c in ROUTERS.values() if c.__module__ == mod.__name__]
        if not found:
            raise KeyError(f"{spec} registers no router; add @register_router to its class")
        return found[-1]
    n = _key(spec)
    if n not in ROUTERS:
        near = difflib.get_close_matches(n, list(ROUTERS), n=1)
        raise KeyError(f"unknown router {spec!r}{f' (did you mean {near[0]!r}?)' if near else ''}; available: "
                       f"{', '.join(sorted(ROUTERS))}, or a file my_router.py")
    return ROUTERS[n]


def get_serving(spec: str):
    """A Serving class by name, or the backend a plugin file registers (my_backend.py)."""
    _builtins()
    if spec.endswith(".py"):
        mod = load_plugin(spec)
        found = [c for c in SERVING.values() if c.__module__ == mod.__name__]
        if not found:
            raise KeyError(f"{spec} registers no serving backend; add @register_serving to its class")
        return found[-1]
    n = _key(spec)
    if n not in SERVING:
        near = difflib.get_close_matches(n, list(SERVING), n=1)
        raise KeyError(f"unknown serving backend {spec!r}{f' (did you mean {near[0]!r}?)' if near else ''}; "
                       f"available: {', '.join(sorted(SERVING))}, or a file my_backend.py")
    return SERVING[n]


def list_routers() -> list:
    _builtins()
    return sorted(ROUTERS)


def list_serving() -> list:
    _builtins()
    return sorted(SERVING)
