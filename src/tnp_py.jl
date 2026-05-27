
###########################################
### The main function to load the model ###
###########################################

function TNP()
    return BlackboxModel(
        blackbox_predict,
    )
end


####################
### ... code ... ###
####################

function blackbox_predict(x::AbstractVector{<:Real}, data::ExperimentData)
    ydim = size(data.Y, 1)

    yt_mean = zeros(ydim)
    yt_std = zeros(ydim)
    for d in 1:ydim
        μ, σ = tnp_predict(x, data.X, @view data.Y[d:d, :])
        yt_mean[d] = μ[1,1,1]  # length(μ) == 1
        yt_std[d] = σ[1,1,1]  # length(σ) == 1
    end
    return yt_mean, yt_std
end
function blackbox_predict(X::AbstractMatrix{<:Real}, data::ExperimentData)
    ydim = size(data.Y, 1)
    n = size(X, 2)

    yt_mean = zeros(ydim, n)
    yt_std = zeros(ydim, n)
    for d in 1:ydim
        μ, σ = tnp_predict(X, data.X, @view data.Y[d:d, :])
        yt_mean[d,:] = μ[1,:,1]  # length(μ) == 1
        yt_std[d,:] = σ[1,:,1]  # length(σ) == 1
    end
    return yt_mean, yt_std
end

function tnp_init()
    # settings
    # TODO change the used model here
    py"""
    import os
    import os.path as osp

    model_name = "tnpd"

    ### 1D
    # model_root = '/home/soldasim/repos/TNP-pytorch/regression'
    # model_weights = osp.join(model_root, f'results/gp/tnpd/default/ckpt.tar')
    # dim_x = 1

    ### 2D
    model_root = '/home/soldasim/repos/TNP-pytorch/bayesian_optimization'
    # model_weights = osp.join(model_root, f'results/highdim_gp/2D/tnpd/min1_max128_10_v1/ckpt.tar') # original
    model_weights = osp.join(model_root, f'results/highdim_gp/2D/tnpd/min1_max128_10_v2/ckpt.tar') # low train data noise
    dim_x = 2
    """

    # add necessary python paths
    py"""
    import sys

    if model_root not in sys.path:
        sys.path.insert(0, model_root)
    
    # from data.gp import *
    """

    # import TNP stuff
    py"""
    # import argparse
    import yaml
    import torch
    import numpy as np
    # import time
    # import matplotlib.pyplot as plt
    # import uncertainty_toolbox as uct
    # from attrdict import AttrDict
    # from tqdm import tqdm
    # from copy import deepcopy
    # import h5py

    from utils.misc import load_module
    # from utils.paths import results_path, evalsets_path, datasets_path
    # from utils.log import get_logger, RunningAverage
    """

    # init model
    py"""
    model_cls = getattr(load_module(osp.join(model_root, f'models/{model_name}.py')), model_name.upper())
    with open(osp.join(model_root, f'configs/gp/{model_name}.yaml'), 'r') as f:
        config = yaml.safe_load(f)
        config['dim_x'] = dim_x

    # if model_name in ["np", "anp", "cnp", "canp", "bnp", "banp", "tnpd", "tnpa", "tnpnd"]:
    model = model_cls(**config)
    
    model.cuda()
    model.eval()

    ### code optimization
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    """

    # load pretrained weights
    py"""
    ckpt = torch.load(model_weights, map_location='cuda')
    model.load_state_dict(ckpt.model)
    """

    # define the predict function
    ### code optimization
    py"""
    _xc = None
    _yc = None
    _xt = None

    def _set_inputs(xc, yc, xt):
        global _xc, _yc, _xt
        _xc = torch.as_tensor(xc, device="cuda")
        _yc = torch.as_tensor(yc, device="cuda")
        _xt = torch.as_tensor(xt, device="cuda")

    def _predict_cached():
        with torch.inference_mode():
            out = model.predict(_xc, _yc, _xt)
        yt_mean = out.loc.cpu().numpy()
        yt_scale = out.scale.cpu().numpy()
        return yt_mean, yt_scale
    """

    # define the predict function with type conversions
    # py"""
    # def _predict(model, xc, yc, xt):
    #     print("evaluating at ", xt) # TODO rem

    #     ### code optimization
    #     # with torch.no_grad():
    #     with torch.inference_mode():
    #         out = model.predict(xc, yc, xt)
        
    #     # convert to numpy
    #     yt_mean = out.loc.cpu().numpy()
    #     yt_std = out.scale.cpu().numpy()

    #     return yt_mean, yt_std
    # """

    return (
        model_name = py"model_name",
        model_weights = py"model_weights",
    )
end

function tnp_predict(Xt::AbstractArray{<:Real}, X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real})
    @assert size(Xt, 1) == size(X, 1)
    @assert size(X, 2) == size(Y, 2)
    x_dim = size(X, 1)
    y_dim = size(Y, 1)
    context_size = size(X, 2)

    # reshape inputs
    xt = zeros(Float32, 1, size(Xt, 2), size(Xt, 1))
    for i in 1:size(Xt, 2)
        xt[1, i, :] .= Xt[:, i]
    end

    xc = zeros(Float32, 1, context_size, x_dim)
    yc = zeros(Float32, 1, context_size, y_dim)
    for i in 1:context_size
        xc[1, i, :] .= X[:, i]
        yc[1, i, :] .= Y[:, i]
    end

    # # send to gpu in python
    # xc = py"torch.tensor"(xc; device="cuda")
    # yc = py"torch.tensor"(yc; device="cuda")
    # xt = py"torch.tensor"(xt; device="cuda")
    
    # # evaluate
    # yt_mean, yt_std = py"_predict"(py"model", xc, yc, xt)
    
    py"_set_inputs"(xc, yc, xt)
    yt_mean, yt_std = py"_predict_cached"()
    
    return yt_mean, yt_std
end

function tnp_reshape_input(arr::AbstractArray)
    arr_ = zeros(Float32, reverse(size(arr)))

    for batch in 1:size(arr, 3)
        for point in 1:size(arr, 2)
            arr_[batch, point, :] .= arr[:, point, batch]
        end
    end

    return arr_
end


###########################
### RUN THIS ON INCLUDE ###
###########################

function init_python()
    if isdefined(Main, :TNP_INITIALIZED)
        @warn "Python already initialized, skipping ..."
        return 
    end

    ### code optimization
    ENV["OMP_NUM_THREADS"] = "1"
    ENV["MKL_NUM_THREADS"] = "1"
    ENV["OPENBLAS_NUM_THREADS"] = "1"
    
    # initialize python only once
    @warn "Initiating Python via PyCall ..."
    info = tnp_init()
    @info "Loaded model: $(info.model_name)"
    @info "  with weights from: $(info.model_weights)"

    # define the flag
    global TNP_INITIALIZED = true
end

init_python() ### run on include
