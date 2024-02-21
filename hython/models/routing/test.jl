#using Pkg
#Pkg.develop("GraphNeuralNetworks")

using GraphNeuralNetworks
using Graphs, Wflow, GraphMakie, GLMakie, FilePaths, GraphIO, Flux, CUDA, MLUtils
using NCDatasets, Dates
using CommonDataModel: @select


# IO https://github.com/JuliaGraphs/GraphIO.jl

# graph neural network https://carlolucibello.github.io/GraphNeuralNetworks.jl/dev/


path = "data/wflow"

config = Wflow.Config(joinpath(path, "wflow_sbm.toml") )

wflow = Wflow.initialize_sbm_model(config)

wrgraph = wflow.network.river.graph 


# adjacency matrix 

adj_matrix = Graphs.adjacency_matrix(wrgraph)

# adjacency list

#Graphs.adjacency_spectrum(wrgraph)


# run model 

model_run = Wflow.run(wflow)


# target 

target = Wflow.NCDataset(joinpath(path, "run_default/output.nc"))["q_river"]

start  = coord(target, "time")[:][begin]

target = convert.(Float32, coalesce.(target, 0))


ds = Wflow.NCDataset(joinpath(path, "forcings.nc"))

feat_precip = @select(ds["precip"], time .>= DateTime(2019,1,2))[wflow.network.river.indices, :]

feat_precip = convert.(Float32, coalesce.(feat_precip, 0))



# feature matrix 

feat_bd = wflow.lateral.river.bankfull_depth
feat_man = wflow.lateral.river.n

feat_bd = repeat(feat_bd, 1, size(feat_precip)[end])

feat_man  = repeat(feat_man , 1, size(feat_precip)[end])

feat_matrix = cat(feat_bd,feat_man,feat_precip,dims=3)

size(feat_matrix)

# Graph neural network object
gnn_graph = GNNGraph(adj_matrix) #, ndata=reshape(feat_matrix, (3, 91, 364)))



#q_river[model.network.river.indices, 1]

#sum(.!ismissing.(q_river[:,:,3]))

# temporal graph https://carlolucibello.github.io/GraphNeuralNetworks.jl/dev/temporalgraph/

#snapshots = [ GNNGraph(adj_matrix, ndata=q_river[model.network.river.indices, t] ) for t in 1:364]
#tg = TemporalSnapshotsGNNGraph(snapshots)


# model run 

device = CUDA.functional() ? Flux.gpu : Flux.cpu;

# model = GNNChain(GCNConv(16 => 64),
#                 BatchNorm(64),     # Apply batch normalization on node features (nodes dimension is batch dimension)
#                 x -> relu.(x),     
#                 GCNConv(64 => 64, relu),
#                 GlobalPool(mean),  # aggregate node-wise features into graph-wise features
#                 Dense(64, 1)) |> device




model = GNNChain(TGCN(3 => 100), Dense(100, 1))


feat_matrix = convert.(Float32,reshape(feat_matrix, (3, 91, 364)))

target = convert.(Float32, reshape(unsqueeze(target,2), (1, 91, 364)))


train_loader = zip(feat_matrix[:,:70,:], target[:,:70,:])
test_loader = zip(feat_matrix[:, 70:end, :], target[:, 70:end, :])


using Flux.Losses: mae

function train(graph, train_loader, model)

    opt = Flux.setup(Adam(0.001), model)

    for epoch in 1:100
        for (x, y) in train_loader
            x, y = (x, y)
            grads = Flux.gradient(model) do model
                ŷ = model(graph, x)
                Flux.mae(ŷ, y) 
            end
            Flux.update!(opt, model, grads[1])
        end
        
        if epoch % 10 == 0
            loss = mean([Flux.mae(model(graph,x), y) for (x, y) in train_loader])
            @show epoch, loss
        end
    end
    return model
end


train(gnn_graph, train_loader, model)









# example

using GraphNeuralNetworks
using Flux
using Flux.Losses: mae
using MLDatasets: METRLA
using Statistics
using Plots

dataset_metrla = METRLA(; num_timesteps = 3)



g = dataset_metrla[1]


size(g.node_data.features[1])

size(g.node_data.targets[1])

g.node_data.features[1][:,1,:]
g.node_data.features[2][:,1,:]
g.node_data.targets[1][:,1,:]

graph = GNNGraph(g.edge_index; edata = g.edge_data, g.num_nodes)
features = g.node_data.features
targets = g.node_data.targets


train_loader = zip(features[1:200], targets[1:200])
test_loader = zip(features[2001:2288], targets[2001:2288])


model = GNNChain(TGCN(2 => 100), Dense(100, 1))


train(graph, train_loader, model)