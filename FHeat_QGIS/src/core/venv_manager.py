import os
import sys
import subprocess
from typing import Optional, Callable, Tuple

from qgis.core import QgsMessageLog, Qgis

# Plugin root: FHeat_QGIS/src/core/venv_manager.py -> go up three levels
PLUGIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PYTHON_VERSION = f"py{sys.version_info.major}.{sys.version_info.minor}"
VENV_DIR = os.path.join(PLUGIN_DIR, f"venv_{PYTHON_VERSION}")

# Aus deiner requirements.txt
REQUIRED_PACKAGES = [
    "geopandas",
    "OWSLib",
    "pandas",
    "fiona",
    "numpy",
    "networkx",
    "matplotlib",
    "openpyxl",
    "demandlib",
    "workalendar",
]


def _log(msg: str, level=Qgis.Info):
    QgsMessageLog.logMessage(msg, "F|Heat", level=level)


def get_venv_python() -> str:
    if sys.platform == "win32":
        return os.path.join(VENV_DIR, "Scripts", "python.exe")
    bin_dir = os.path.join(VENV_DIR, "bin")
    for name in (
        f"python{sys.version_info.major}.{sys.version_info.minor}",
        f"python{sys.version_info.major}",
        "python3",
        "python",
    ):
        candidate = os.path.join(bin_dir, name)
        if os.path.exists(candidate):
            return candidate
    # venv not yet created — preferred name as fallback (used only for existence checks)
    return os.path.join(bin_dir, f"python{sys.version_info.major}.{sys.version_info.minor}")


def get_venv_site_packages() -> str:
    if sys.platform == "win32":
        return os.path.join(VENV_DIR, "Lib", "site-packages")
    lib = os.path.join(VENV_DIR, "lib")
    if os.path.exists(lib):
        for d in os.listdir(lib):
            if d.startswith("python"):
                sp = os.path.join(lib, d, "site-packages")
                if os.path.exists(sp):
                    return sp
    return os.path.join(lib, f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages")


def venv_exists() -> bool:
    return os.path.exists(get_venv_python())


def _find_python() -> str:
    """Return the real Python interpreter.

    On macOS/Linux, sys.executable is the QGIS app binary, not Python.
    We search sys.exec_prefix/bin/ for a versioned Python binary instead.
    On Windows, sys.executable is qgis.exe — same issue, search exec_prefix.
    """
    ver_name_unix = f"python{sys.version_info.major}.{sys.version_info.minor}"
    ver_name_win  = f"python{sys.version_info.major}{sys.version_info.minor}.exe"

    if sys.platform == "win32":
        for name in (ver_name_win, "python3.exe", "python.exe"):
            for search_dir in (sys.exec_prefix, os.path.dirname(sys.executable)):
                candidate = os.path.join(search_dir, name)
                if os.path.exists(candidate):
                    return candidate
        return sys.executable

    # macOS / Linux: sys.executable may be the QGIS binary, not Python.
    # The real Python lives in sys.exec_prefix/bin/.
    bin_dir = os.path.join(sys.exec_prefix, "bin")
    for name in (ver_name_unix, f"python{sys.version_info.major}", "python3", "python"):
        candidate = os.path.join(bin_dir, name)
        if os.path.exists(candidate):
            _log(f"Python-Interpreter gefunden: {candidate}")
            return candidate

    # Last resort: sys.executable only if it looks like Python
    if "python" in os.path.basename(sys.executable).lower():
        return sys.executable

    raise RuntimeError(
        f"Kein Python-Interpreter in {bin_dir} gefunden. "
        "sys.executable ist: " + sys.executable
    )


def _clean_env() -> dict:
    from .subprocess_utils import get_clean_env_for_venv
    return get_clean_env_for_venv()


def _subprocess_kwargs() -> dict:
    from .subprocess_utils import get_subprocess_kwargs
    return get_subprocess_kwargs()


def create_venv(progress_callback: Optional[Callable] = None) -> Tuple[bool, str]:
    _log(f"Erstelle venv: {VENV_DIR}")
    if progress_callback:
        progress_callback(5, "Erstelle virtuelle Umgebung...")
    try:
        host_python = _find_python()
    except RuntimeError as e:
        return False, str(e)
    _log(f"Verwende Python: {host_python}")
    result = subprocess.run(
        [host_python, "-m", "venv", VENV_DIR, "--without-pip"],
        capture_output=True, text=True,
        env=_clean_env(), **_subprocess_kwargs()
    )
    if result.returncode != 0:
        return False, f"venv-Erstellung fehlgeschlagen: {result.stderr}"
    # After venv creation, resolve the actual python binary inside the venv
    python = get_venv_python()
    if not os.path.exists(python):
        return False, f"venv erstellt, aber Python-Binary nicht gefunden: {python}"
    # Bootstrap pip (QGIS Python often ships without ensurepip/pip in venv)
    if progress_callback:
        progress_callback(10, "Bootstrap pip...")
    pip_result = subprocess.run(
        [python, "-m", "ensurepip", "--upgrade"],
        capture_output=True, text=True,
        env=_clean_env(), **_subprocess_kwargs()
    )
    if pip_result.returncode != 0:
        _log(f"ensurepip fehlgeschlagen, lade get-pip.py herunter: {pip_result.stderr}", Qgis.Warning)
        import urllib.request, tempfile
        get_pip_path = os.path.join(tempfile.gettempdir(), "get-pip.py")
        try:
            urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py", get_pip_path)
        except Exception as e:
            return False, f"get-pip.py konnte nicht heruntergeladen werden: {e}"
        bootstrap_result = subprocess.run(
            [python, get_pip_path],
            capture_output=True, text=True,
            env=_clean_env(), **_subprocess_kwargs()
        )
        if bootstrap_result.returncode != 0:
            return False, f"pip-Bootstrap fehlgeschlagen: {bootstrap_result.stderr}"
    # Upgrade pip to latest
    subprocess.run(
        [python, "-m", "pip", "install", "--upgrade", "pip"],
        capture_output=True, text=True,
        env=_clean_env(), **_subprocess_kwargs()
    )
    return True, "Virtuelle Umgebung erstellt"


def install_packages(progress_callback: Optional[Callable] = None) -> Tuple[bool, str]:
    python = get_venv_python()
    env = _clean_env()
    total = len(REQUIRED_PACKAGES)
    for i, pkg in enumerate(REQUIRED_PACKAGES):
        if progress_callback:
            pct = 20 + int((i / total) * 75)
            progress_callback(pct, f"Installiere {pkg}...")
        result = subprocess.run(
            [python, "-m", "pip", "install", pkg],
            capture_output=True, text=True,
            env=env, **_subprocess_kwargs()
        )
        if result.returncode != 0:
            err = result.stderr or result.stdout
            _log(f"Fehler bei {pkg}: {err}", Qgis.Warning)
            return False, f"Installation von '{pkg}' fehlgeschlagen: {err}"
    if progress_callback:
        progress_callback(100, "Installation abgeschlossen")
    return True, "Pakete installiert"


def packages_installed() -> bool:
    """Return True if venv exists and all REQUIRED_PACKAGES are importable from it."""
    if not venv_exists():
        return False
    python = get_venv_python()
    check = "; ".join(f"import importlib; importlib.import_module('{p.lower().replace('-','_')}')" for p in REQUIRED_PACKAGES)
    result = subprocess.run(
        [python, "-c", check],
        capture_output=True, text=True,
        env=_clean_env(), **_subprocess_kwargs()
    )
    return result.returncode == 0


def create_venv_and_install(progress_callback: Optional[Callable] = None) -> Tuple[bool, str]:
    ok, msg = create_venv(progress_callback)
    if not ok:
        return False, msg
    return install_packages(progress_callback)


def ensure_packages_available() -> bool:
    """Fügt venv site-packages zu sys.path hinzu."""
    if not venv_exists():
        return False
    sp = get_venv_site_packages()
    if sp and sp not in sys.path:
        sys.path.insert(0, sp)
        _log(f"venv site-packages hinzugefügt: {sp}")
    return True