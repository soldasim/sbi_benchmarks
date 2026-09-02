#!/bin/sh

### README ###
# Follow these steps:
# 1) conda activate simformer
# 2) sh julia_simformer.sh

# source ~/.bashrc
julia -e """
    python_path = \"$(which python)\"
    if !contains(python_path, \"simformer\")
        @error \"Activate the python environment first: \`conda activate simformer\`\"
        exit()
    end
    
    ENV[\"PYTHON\"] = python_path
    ENV[\"JULIA_CONDAPKG_BACKEND\"] = \"Null\"
    ENV[\"JULIA_PYTHONCALL_EXE\"] = python_path
    using Pkg
    Pkg.activate(\"src\")
    using PythonCall
""" -i
