# -*- mode: python ; coding: utf-8 -*-
# Construction : voir scripts/build_exe.ps1 (dist et build hors OneDrive).
from PyInstaller.utils.hooks import collect_all, collect_submodules, copy_metadata

datas, binaries, hiddenimports = [], [], []
# ChromaDB, fastembed et onnxruntime chargent des modules et des DLL dynamiquement.
for package in ('chromadb', 'fastembed', 'onnxruntime', 'tokenizers'):
    d, b, h = collect_all(package)
    datas += d
    binaries += b
    hiddenimports += h
hiddenimports += collect_submodules('uvicorn')
# mcp.cli requiert typer (extra mcp[cli]) et ne sert pas au serveur
hiddenimports += collect_submodules('mcp', filter=lambda name: not name.startswith('mcp.cli'))
for package in ('mcp', 'chromadb', 'fastembed', 'onnxruntime', 'tokenizers', 'huggingface_hub', 'tqdm'):
    datas += copy_metadata(package)

a = Analysis(
    ['scripts/exe_entry.py'],
    pathex=['src'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['pytest', 'tkinter'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='episodic-memory-mcp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='episodic-memory-mcp',
)
