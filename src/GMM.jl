module GMM

export gmm_spec, gmm, reg, reg_table, mv_reg, principal_components, exact, forwarddiff, nw, hh, white, preset

include("f0_header.jl")
include("f1_gmm.jl")
include("f2_options.jl")
include("f3_reg.jl")

end # module
