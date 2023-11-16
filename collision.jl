using Plots
include("reach.jl")


# returns H-rep of input set
function input_constraints_collision()
    # lb: [5, 5, -1, -1, 0, 0, -1, -1]
    # ub: [6, 6, 1, 1, 10, 10, 1, 1]
    # input constraint in the form Ax<=b
    A = [
        1. 0;
        -1 0;
        1. 0;
        -1 0;
        1. 0;
        -1 0;
        1. 0;
        -1 0;
        1. 0;
        -1 0;
        1. 0;
        -1 0;
        1. 0;
        -1 0;
        1. 0;
        -1 0
    ]
    b = [6 -5 6 -5 1 1 1 1 10 0 10 0 1 1 1 1]

    return A, b
end


# returns H-rep of output sets
function output_constraints_collision()
    A = [
        1 0 0 0 -1 0 0 0;
        0 1 0 0 0 -1 0 0;
        -1 0 0 0 1 0 0 0;
        0 -1 0 0 0 1 0 0
    ]
    b = [1 1 1 1]

    return A, b
end


function load_torch_model(model)
    W = npzread(model)
    layer_sizes = numpy.array()



###########################
######## SCRIPTING ########
###########################
copies = 1 # copies = 1 is original network
model = "models/Collision/merged.npy"
weights = pytorch_net(model, copies)

A_in, B_in = input_constraints_collision()
A_out, B_out = output_constraints_collision()

# Run algorithm
@time begin
ap2input, ap2output, ap2map, ap2backward = compute_reach(weights, A_in, B_in, [A_out], [B_out], reach=true, back=true, verification=true)

end
@show length(ap2input)
