using BOSS, BOSIP

f_constant_1d(x) = 0.
f_smooth_1d(x) = x[1]

function test_predict_1d()
    # TODO
    obj = f_constant_1d
    # obj = f_smooth_1d

    bounds = ()

    ## Simulator
    function sim(x)
        y = obj(x)
        return [y]
    end


end
