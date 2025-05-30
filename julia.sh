# something to avoid segmentation faults in juliacall
export PYTHON_JULIACALL_HANDLE_SIGNALS=yes

# activate python env
. venv/bin/activate

# start julia
julia -e "

    # activate julia env
    using Pkg
    Pkg.activate(\"venv/julia_env/\")

    # set PyCall.jl python path
    ENV[\"PYCALL_JL_RUNTIME_PYTHON\"] = Sys.which(\"python\")
    using PyCall
    @info \"PyCall python path:     \" * PyCall.pyimport(\"sys\").executable
    
    # set PythonCall.jl python path
    ENV[\"JULIA_CONDAPKG_BACKEND\"] = \"Null\"
    ENV[\"JULIA_PYTHONCALL_EXE\"] = \"@PyCall\"
    using PythonCall
    @info \"PythonCall python path: \" * pyconvert(String, PythonCall.pyimport(\"sys\").executable)

    # import juliacall
    PyCall.pyimport(\"juliacall\")

    # start interactive REPL
" -i
