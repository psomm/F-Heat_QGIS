import os
import shutil
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


def _check_python_candidate(candidate: str) -> bool:
    """Return True if *candidate* exists, runs, and matches the current major.minor version."""
    if not os.path.isfile(candidate):
        return False
    expected = f"{sys.version_info.major}.{sys.version_info.minor}"
    try:
        result = subprocess.run(
            [candidate, "-c",
             "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0 and result.stdout.strip() == expected
    except Exception:
        return False


def _find_macos_qgis_pythonhome(host_python: str) -> Optional[str]:
    """Return PYTHONHOME for a QGIS-bundled Python on macOS, or *None*.

    If *host_python* lives inside a ``*.app`` bundle, look for the
    ``Python.framework`` directory inside that bundle and return the
    versioned prefix path (e.g.
    ``/Applications/QGIS.app/Contents/Frameworks/Python.framework/Versions/3.12``).
    """
    real_path = os.path.realpath(host_python)
    parts = real_path.split(os.sep)
    app_idx = next((i for i, p in enumerate(parts) if p.endswith(".app")), None)
    if app_idx is None:
        return None

    app_contents = os.sep.join(parts[: app_idx + 1]) + os.sep + "Contents"
    ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    candidates = [
        os.path.join(app_contents, "Frameworks", "Python.framework", "Versions", ver),
        os.path.join(app_contents, "Frameworks", "Python.framework", "Versions", "Current"),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            _log(f"macOS QGIS-Python: PYTHONHOME → {candidate}")
            return candidate
    return None


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

    # macOS: prefer an external (system / Homebrew / framework) Python that can
    # actually create venvs, before falling back to the QGIS-bundled binary whose
    # build-time prefix paths may be unavailable on the user's machine.
    if sys.platform == "darwin":
        ver = f"{sys.version_info.major}.{sys.version_info.minor}"
        macos_search = [
            # 1. System Python
            "/usr/bin/python3",
            # 2. Homebrew (Apple Silicon)
            f"/opt/homebrew/bin/{ver_name_unix}",
            f"/opt/homebrew/bin/python3",
            # 3. Homebrew (Intel / x86-64)
            f"/usr/local/bin/{ver_name_unix}",
            f"/usr/local/bin/python3",
            # 4. Python.org standalone framework
            f"/Library/Frameworks/Python.framework/Versions/{ver}/bin/python3",
        ]
        # 5. Versioned binary from PATH (e.g. installed by pyenv or conda)
        which_path = shutil.which(ver_name_unix)
        if which_path and which_path not in macos_search:
            macos_search.append(which_path)

        for candidate in macos_search:
            if _check_python_candidate(candidate):
                _log(f"macOS: externer Python-Interpreter gefunden: {candidate}")
                return candidate

        _log(
            "macOS: Kein passender externer Python gefunden – verwende QGIS-Python "
            "(PYTHONHOME wird in create_venv gesetzt).",
            Qgis.Warning,
        )
        # Fall through to QGIS-bundled Python search below.

    # macOS / Linux: sys.executable may be the QGIS binary, not Python.
    # Python can live in different locations depending on the QGIS build:
    #   - sys.exec_prefix/bin/         (e.g. Contents/Frameworks/bin/)
    #   - dirname(sys.executable)/bin/ (e.g. Contents/MacOS/bin/)
    #   - parent of exec_prefix / bin/ (e.g. Contents/bin/)
    search_dirs = []
    for d in (
        os.path.join(sys.exec_prefix, "bin"),
        os.path.join(os.path.dirname(sys.executable), "bin"),
        os.path.join(os.path.dirname(sys.exec_prefix), "bin"),
        os.path.dirname(sys.executable),  # Python directly next to QGIS binary
    ):
        if d not in search_dirs:
            search_dirs.append(d)

    for bin_dir in search_dirs:
        for name in (ver_name_unix, f"python{sys.version_info.major}", "python3", "python"):
            candidate = os.path.join(bin_dir, name)
            if os.path.exists(candidate):
                _log(f"Python-Interpreter gefunden: {candidate}")
                return candidate

    # Last resort: sys.executable only if it looks like Python
    if "python" in os.path.basename(sys.executable).lower():
        return sys.executable

    searched = ", ".join(search_dirs)
    raise RuntimeError(
        f"Kein Python-Interpreter gefunden. Durchsucht: {searched}. "
        f"sys.executable ist: {sys.executable}"
    )


def _clean_env(python_home: Optional[str] = None) -> dict:
    from .subprocess_utils import get_clean_env_for_venv
    return get_clean_env_for_venv(python_home=python_home)


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

    # On macOS, if the selected Python lives inside a .app bundle its
    # build-time stdlib paths may not exist on the user's machine.  Set
    # PYTHONHOME so the subprocess can locate the standard library.
    python_home: Optional[str] = None
    if sys.platform == "darwin":
        python_home = _find_macos_qgis_pythonhome(host_python)
        if python_home:
            _log(f"macOS: PYTHONHOME für venv-Erstellung → {python_home}")

    result = subprocess.run(
        [host_python, "-m", "venv", VENV_DIR, "--without-pip"],
        capture_output=True, text=True,
        env=_clean_env(python_home), **_subprocess_kwargs()
    )

    # On Linux (and macOS when no .app-bundle PYTHONHOME was found), a
    # QGIS-bundled Python (AppImage, Flatpak, …) needs PYTHONHOME to locate
    # its standard library.  Detect this by the well-known error message and
    # retry with sys.exec_prefix as PYTHONHOME.
    if (
        result.returncode != 0
        and "platform independent libraries" in result.stderr
        and python_home is None
        and sys.platform != "win32"
    ):
        python_home = sys.exec_prefix
        _log(
            f"venv-Erstellung fehlgeschlagen (Bibliotheken nicht gefunden). "
            f"Wiederhole mit PYTHONHOME={python_home}",
            Qgis.Warning,
        )
        # Remove any partial venv directory left by the failed attempt.
        if os.path.exists(VENV_DIR):
            try:
                shutil.rmtree(VENV_DIR)
            except OSError as exc:
                _log(f"Warnung: Konnte unvollständiges venv nicht löschen: {exc}", Qgis.Warning)
        result = subprocess.run(
            [host_python, "-m", "venv", VENV_DIR, "--without-pip"],
            capture_output=True, text=True,
            env=_clean_env(python_home), **_subprocess_kwargs()
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