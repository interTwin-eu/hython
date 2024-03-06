using Wflow: kinematic_wave
using Plots

"""

From Chow Applied Hydrology 1988

α = [n*P**(2/3) / (1.49*√(Sₒ)) ]**0.6 
β = 0.6 

P = wetted perimeter 
Sₒ = bottom river slope 
n = manning roughness

Kinematic Wave:

∂A/∂t + ∂Q/∂x = q 
Sf = So 

A = αQᵝ

In Surface Flow River: 

β::T | "-" | 0 | "scalar"                       # constant in Manning's equation
sl::Vector{T} | "m m-1"                         # Slope [m m⁻¹]
n::Vector{T} | "s m-1/3"                        # Manning's roughness [s m⁻⅓]
dl::Vector{T} | "m"                             # Drain length [m]
q::Vector{T} | "m3 s-1"                         # Discharge [m³ s⁻¹]
qin::Vector{T} | "m3 s-1"                       # Inflow from upstream cells [m³ s⁻¹]
q_av::Vector{T} | "m3 s-1"                      # Average discharge [m³ s⁻¹]
qlat::Vector{T} | "m2 s-1"                      # Lateral inflow per unit length [m² s⁻¹]
inwater::Vector{T} | "m3 s-1"                   # Lateral inflow [m³ s⁻¹]
volume::Vector{T} | "m3"                        # Kinematic wave volume [m³] (based on water level h)
h::Vector{T} | "m"                              # Water level [m]
h_av::Vector{T} | "m"                           # Average water level [m]
Δt::T | "s" | 0 | "none" | "none"               # Model time step [s]
its::Int | "-" | 0 | "none" | "none"            # Number of fixed iterations
width::Vector{T} | "m"                          # Flow width [m]
alpha_pow::T | "-" | 0 | "scalar"               # Used in the power part of α
alpha_term::Vector{T} | "-"                     # Term used in computation of α
α::Vector{T} | "s3/5 m1/5"                      # Constant in momentum equation A = αQᵝ, based on Manning's equation
cel::Vector{T} | "m s-1"                        # Celerity of the kinematic wave
to_river::Vector{T} | "m3 s-1"                  # Part of overland flow [m³ s⁻¹] that flows to the river
kinwave_it::Bool | "-" | 0 | "none" | "none"    # Boolean for iterations kinematic wave


sf.q[v] = kinematic_wave(
    sf.qin[v],
    sf.q[v],
    sf.qlat[v] + inflow,
    sf.α[v],
    sf.β,
    Δt,
    sf.dl[v]
    )


"""

Δt = 60 * 60 * 24 # s
Δx = 100_000  # m 

# discretization 

nt = 31
it = collect(range(1,nt))

qin = 1*(10 .+ sin.(it)) 
q = repeat([1.0], nt)
qlat =  repeat([1.0], nt) / Δx
alpha = 0.1
beta =  0.6

qout = kinematic_wave.(qin, q, qlat, alpha, beta, Δt, Δx)

plot(qin, label="qin", xlabel="Δt", ylabel = "Q (m³s⁻¹)")
plot!(qout, label="qout")


# chain multiple river reaches