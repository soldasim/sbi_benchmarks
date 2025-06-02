# something to avoid segmentation faults in juliacall
export PYTHON_JULIACALL_HANDLE_SIGNALS=yes

# activate python env
. venv/bin/activate

# start julia
julia -e "
    include(\"activate.jl\")
" -i
