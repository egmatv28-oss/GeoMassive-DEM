"""ГЕОМАССИВ/2D · DEM — точка входа.

Решатель — в solver.py, графика отображение — в render.py.

Запуск:  python mineudec.py [--selftest]
"""

import os
import subprocess
import sys

JUNCTION = r"C:\py312"
JUNCTION_EXE = JUNCTION + r"\python.exe"


def _is_ascii(s):
    return all(ord(c) < 128 for c in s)


def _taichi_home():
    """Путь до каталога taichi в site-packages — без импорта самого taichi.

    Импорт taichi в родительском процессе заставляет его захватывать lock
    кеша компиляции (ticache.lock), который потом не может взять дочерний
    процесс — отсюда предупреждение "[W] Lock ... failed". Чтобы родитель не
    создавал лишний KernelCompilationManager, ограничиваемся поиском пути.
    """
    for p in sys.path:
        if p and "site-packages" in p.replace("\\", "/").lower():
            return p
    return None


def _needs_relaunch():
    home = _taichi_home()
    try:
        return home is not None and not _is_ascii(home)
    except Exception:
        return False


def _ensure_junction():
    if os.path.exists(JUNCTION_EXE):
        return True
    target = os.path.dirname(sys.executable)
    if _is_ascii(target):
        return False
    try:
        import ctypes
        import ctypes.wintypes as w
        from ctypes import wintypes

        class REPARSE_DATA_BUFFER(ctypes.Structure):
            _fields_ = [("ReparseTag", w.DWORD), ("ReparseDataLength", w.WORD),
                        ("Reserved", w.WORD), ("SubstituteNameOffset", w.WORD),
                        ("SubstituteNameLength", w.WORD), ("PrintNameOffset", w.WORD),
                        ("PrintNameLength", w.WORD), ("Flags", w.DWORD),
                        ("PathBuffer", ctypes.c_char * 512)]

        CreateSymbolicLink = ctypes.windll.kernel32.CreateSymbolicLinkW
        CreateSymbolicLink.restype = w.BOOL
        CreateSymbolicLink.argtypes = [w.LPCWSTR, w.LPCWSTR, w.DWORD]
        ok = CreateSymbolicLink(JUNCTION, target, 1)  # 1 = directory
        return bool(ok) and os.path.exists(JUNCTION_EXE)
    except Exception:
        return False


def main():
    args = sys.argv[1:]
    if "--selftest" in args:
        import render
        render.main()
        return
    if _needs_relaunch():
        if _ensure_junction() and os.path.exists(JUNCTION_EXE):
            sys.exit(subprocess.call([JUNCTION_EXE, os.path.abspath(__file__)] + args))
        print(
            "Taichi не может открыть свои шейдеры: путь до site-packages содержит кириллицу.\n"
            "Скопируйте Python в путь без кириллицы либо создайте junction C:\\py312 -> Python312\n"
            "и запускайте:  C:\\py312\\python.exe mineudec.py",
            file=sys.stderr,
        )
        sys.exit(1)
    import render
    render.main()


if __name__ == "__main__":
    main()
