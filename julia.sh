# something to avoid segmentation faults in juliacall
export PYTHON_JULIACALL_HANDLE_SIGNALS=yes

# activate python env
ml Python/3.10 # for cluster
. venv/bin/activate

# start julia
julia -e "
    using Revise
    include(\"activate.jl\")
" -i
