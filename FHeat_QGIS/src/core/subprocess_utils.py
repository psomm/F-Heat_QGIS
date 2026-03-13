import os
import sys
import subprocess
from typing import Optional


def get_clean_env_for_venv(python_home: Optional[str] = None) -> dict:
    """Saubere Umgebung für venv-Subprozesse (kein QGIS-Leak).

    Parameters
    ----------
    python_home:
        Optional path to set as PYTHONHOME.  When provided (e.g. to point a
        QGIS-bundled Python at its own framework on macOS) the variable is set
        instead of being stripped.
    """
    env = os.environ.copy()
    for var in ['PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV',
                'QGIS_PREFIX_PATH', 'QGIS_PLUGINPATH']:
        env.pop(var, None)
    if python_home:
        env['PYTHONHOME'] = python_home
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def get_subprocess_kwargs() -> dict:
    """Windows: kein sichtbares cmd-Fenster."""
    kwargs = {}
    if sys.platform == "win32":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        kwargs['startupinfo'] = startupinfo
        kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
    return kwargs