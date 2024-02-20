using Graphs, Wflow, GraphNeuralNetworks, GraphMakie, GLMakie, FilePaths, GraphIO

# IO https://github.com/JuliaGraphs/GraphIO.jl

# graph neural network https://carlolucibello.github.io/GraphNeuralNetworks.jl/dev/


path = "../../../data/wflow"

config = Wflow.Config(joinpath(path, "wflow_sbm.toml") )

model = Wflow.initialize_sbm_model(config)

wrgraph = model.network.river.graph 


# adjacency matrix 

adj_matrix = Graphs.adjacency_matrix(wrgraph)

# adjacency list

#Graphs.adjacency_spectrum(wrgraph)

# feature matrix 

feat_bd = model.lateral.river.bankfull_depth
feat_man = model.lateral.river.mannings_n

feat_matrix = cat(feat_bd,feat_man,dims=2)


# Graph neural network object
gnn_graph = GNNGraph(adj_matrix, ndata=transpose(feat_matrix))


# run model 

model_run = Wflow.run(model)


q_river = Wflow.NCDataset(joinpath(path, "run_default/output.nc"))["q_river"]

q_river[model.network.river.indices, 1]

sum(.!ismissing.(q_river[:,:,3]))

# temporal graph https://carlolucibello.github.io/GraphNeuralNetworks.jl/dev/temporalgraph/

snapshots = [ GNNGraph(adj_matrix, edata=q_river[model.network.river.indices, t] ) for t in 1:13]


tg = TemporalSnapshotsGNNGraph(snapshots)


tg.num_nodes
