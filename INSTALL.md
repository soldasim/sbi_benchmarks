
This repo contains implementation of a wrapper of BOLFI.jl for the `sbibm` benchmark.

# Installation

The installation and the startup are quite complex due to using PyCall.jl and PythonCall.jl simultaneously. PythonCall/JuliaCall are neede for some SBIBM tasks. PyCall.jl is much nicer to use for me.

- `pyenv install 3.10.16`
- `pyenv local 3.10.16`
- `python -m venv venv`
- `. venv/bin/activate`
- `pip install -r requirements.txt`
- manually edit `/venv/lib/python3.10/site-packages/diffeqtorch/diffeqtorch.py` - see the TODOs
- python: `import juliacall` -> creates `/venv/julia_env` used by PythonCall/JuliaCall
- (use the created `/venv/julia_env` environment as the main Julia environment for the project)
- `julia`
- `] activate venv/julia_env`
- `] add PyCall, PythonCall, DifferentialEquations`

# Startup (on every run)

Evaluate the script `sh julia.sh`, which will set up everything (including activating all environments) and start a Julia REPL.

Alternatively, do the set up manually, in which case do the following (in this order):
- `export PYTHON_JULIACALL_HANDLE_SIGNALS=yes` - something to avoid segfaults with JuliaCall
- `. venv/bin/activate` - activate python env
- `julia`
- `] activate venv/julia_env` - activate julia env
- `ENV["PYCALL_JL_RUNTIME_PYTHON"] = Sys.which("python")` - set python for PyCall.jl
- `using PyCall` - only then import
- (check `PyCall.pyimport("sys").executable`)
- `ENV["JULIA_CONDAPKG_BACKEND"] = "Null"` - tell PythonCall.jl to not use custom conda env
- `ENV["JULIA_PYTHONCALL_EXE"] = "@PyCall"` - tell PythonCall.jl to use the same python as PyCall.jl
- `using PythonCall`
- (check `PythonCall.pyimport("sys").executable`)
- `PyCall.pyimport(\"juliacall\")` - import juliacall (into python)
