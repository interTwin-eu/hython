using Graphs
using GraphIO
using EzXML
using Wflow, GraphMakie, GLMakie, FilePaths

# GraphNeuralNetworks

using DataStructures

# IO https://github.com/JuliaGraphs/GraphIO.jl

# graph neursal network https://carlolucibello.github.io/GraphNeuralNetworks.jl/dev/

# https://docs.juliahub.com/General/AttributeGraphs/stable/usage/

# https://github.com/JuliaGraphs/MetaGraphs.jl?tab=readme-ov-file

# http://graphml.graphdrawing.org/primer/graphml-primer.html#Attributes

#using MetaGraphs

using AttributeGraphs

path = "/home/iacopo/dev/hython/data/wflow"

config = Wflow.Config(joinpath(path, "wflow_sbm.toml") )

model = Wflow.initialize_sbm_model(config)

wrgraph = model.network.river.graph 

wrgraph

wlgraph = model.network.land.graph 

wlgraph


savegraph("land_graph.ml",  wlgraph, "wflow", GraphIO.GraphML.GraphMLFormat())



savegraph("graph.ml",  wrgraph, "wflow", GraphIO.GraphML.GraphMLFormat())

gmeta = MetaDiGraph(wrgraph)

gmeta = AttributeGraph(wrgraph, (v) -> 1, DefaultDict{Tuple{Int, Int, String},Float64}(10.0), missing)

for e in edges(gmeta)
    edge_attr(gmeta)[src(e), dst(e), "key"] =10
end

edge_attr(gmeta)

savegraph("graph.ml",  gmeta, "wflow", GraphIO.GraphML.GraphMLFormat())

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