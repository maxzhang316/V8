# V8 dependencies — proposed unified environment

The uploaded snapshots show:

| Package | Adam snapshot | Pty-Chi snapshot |
|---|---|---|
| torch | 2.13.0+cu130 | 2.14.0+cu132 |
| torchvision | 0.28.0+cu130 | 0.29.0+cu132 |
| ptychi | not installed | 2.1.0 |
| numpy | 2.4.6 | 2.4.6 (exported as a non-portable file:// URL) |

Do not concatenate the old lock files: PyTorch/CUDA builds conflict, and
`file:///D:/...` entries in the Pty-Chi freeze file cannot be reused by
other people. The provided requirements.txt is a *candidate for a new*
combined environment, **not a verified lock file**. It chooses the observed
Pty-Chi-side torch/torchvision public version numbers because Pty-Chi was
installed there. V8 Adam must be regression-tested on this choice.

## New environment — Windows PowerShell

Run in your V8 project directory. Copy `environment.yml`, `requirements.txt`
and optionally `requirements-extra.txt` to the V8 root.

```powershell
conda env create -f .\environment.yml
conda activate v8-unified
python --version
python -m pip --version
```

Before installing `requirements.txt`, install a CUDA-enabled matching pair
of **torch 2.14.0 / torchvision 0.29.0** for your OS, driver, Python version,
and GPU using a verified wheel source (e.g., the appropriate PyTorch release
instructions). Your existing Pty-Chi snapshot used `+cu132`, but do not
assume a particular CUDA wheel index exists without checking. If the
matching builds are unavailable, do not silently substitute different
public versions; update and re-test the pins instead.

Once `python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"`
reports the intended torch version and `True`, run:

```powershell
python -m pip install -r .\requirements.txt
python -m pip check
python -c "import torch, ptychi, yaml, h5py; print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available()); print('ptychi import OK')"
```

If you need abTEM generation and a Jupyter user guide:

```powershell
python -m pip install -r .\requirements-extra.txt
```

`requirements-extra.txt` contains new dependencies not documented in the
uploaded snapshots; verify them in the new environment and pin them after
successful tests. `torchaudio` was installed in the Adam snapshot but is not
included in core requirements because it is not needed for the described
ptychographic reconstruction; add it only if an actual source import needs
it, and choose a version compatible with the selected torch build.

## Validation before replacing existing environments

1. Keep your two known-working environments. A single source-code folder
   does not require a single Python environment.
2. Confirm all three algorithms import and CUDA is available.
3. Run the same Dataset A/D input and config through Adam, ePIE and LSQML.
4. Compare outputs and computation timings with your previously validated
   baselines; do not compare unequal batch sizes as evidence of speedup.
5. Only after the merged environment passes, export a new lock from it:
   `python -m pip freeze > requirements-unified-lock.txt`
   Inspect this file for `file:///...` or machine-local paths before sharing.

If a unified environment fails, continue to keep the *code* in one V8
folder but launch Pty-Chi runners with their existing separate Python
interpreter. `conda` itself does **not** belong in requirements.txt; it is an
environment manager, not a pip dependency.
