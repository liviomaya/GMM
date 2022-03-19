
"""
`b, bCov, e, R2 = reg(y::Array{Float64,1}, x; constant=true, S=white(), R2adj=false)`

`S` can be `preset(x)` for a given `x`, `nw(k)` or `hh(k)` for given number of lags, or still `white()`

# Asymptotic Distribution

√T (b - b₀) → 𝑁(0, bCov)

"""
function reg(y::Array{Float64,1}, x; constant=true, S=white(), R2adj=false)

    T = size(x, 1)

    # solve E(ϵxᵀ) = 0
    X = constant ? [ones(T) x] : x
    b = inv(X' * X) * X' * y
    e = y .- X * b

    # calculate g, dg/db and S = T Var(g)
    dg = -(X' * X) / T # - E(xₜxₜᵀ)
    function f(D, β)
        y = D[:, 1]
        x = D[:, 2:end]
        T, P = size(x)

        mm = zeros(T, P)
        for t = 1:T
            ee = y[t] .- dot(β, x[t, :])
            mm[t, :] .= x[t, :] * ee
        end
        return mm
    end
    s = S([y X], f, b)

    # asymptotic distributions
    bCov = inv(dg) * s * inv(dg)'
    # R2 = [e, demean(y)] .|> (z -> z.^2) .|> (z -> sum(z, dims=1)) |> (z -> 1 - z[1] / z[2])
    R2 = 1.0 - sum(e .^ 2) / sum(demean(y) .^ 2)
    R2adj && (R2 = 1 - (1 - R2) * (T - 1) / (T - size(x, 2) - 1))

    return b, bCov, e, R2
end

function reg_table(y, x, id::Vector{Vector{Int64}}; constant=true, S=white(), R2adj=false)
    @assert size(y, 2) == 1 "Method available only to single equation model"

    P = size(x, 2)
    C = length(id)
    T = length(y)
    # data = NaN * ones(2 * C, P + 2)
    data::Array{Any} = fill("", 2 * C, P + 2 + constant)
    for (c, comb) in enumerate(id)
        b, bCov, e, R2 = reg(y, x[:, comb], constant=constant, S=S, R2adj=R2adj)
        σb = sqrt.(diag(bCov) / T)
        tb = (b ./ σb)
        p_val = 2 * (1 .- cdf(Normal(0, 1), abs.(tb)))

        I = (1+constant):length(b)
        X2 = T * b[I]' * inv(bCov[I, I]) * b[I] # asymptotically equivalent to J test below 

        #= J test (assumes constant):
            g = ([ones(T) x[:,comb]]' * demean(y) / T) 
            f(z, b) = z[end] * z[1:end - 1]
            s = S([ones(T) x[:,comb] e], f, b)
            J = T * g' * inv(s) * g =#
        Pcomb = length(comb)
        p_val_X2 = 1 .- cdf(Chisq(Pcomb), X2)

        i = 2 * c - 1
        rows = [i; i + 1]
        cols = constant ? [1; comb .+ 1] : comb
        data[rows, cols] = [b'; p_val']
        data[rows, end-1] = [X2; p_val_X2]
        data[rows[1], end] = R2
    end

    header = [["β$(i - constant)" for i = 1:(P+constant)]; "𝜒²(𝛽=0)"; "R-Sq"]
    constant && (header[1] = "α")
    formatters = ((v, i, j) -> ((j == P + 2 + constant) && isodd(i)) ? 100 * v : v, ft_printf("%3.2f", 1:P+1+constant), ft_printf("%3.1f", 1:P+2+constant))
    h_coef = Highlighter((y, i, j) -> isodd(i), foreground=:blue, bold=true)
    h_tstat = Highlighter((y, i, j) -> iseven(i), foreground=:yellow)
    kw = [:header => header, :formatters => formatters, :vlines => [P + 1], :crop => :all, :highlighters => (h_coef, h_tstat)]
    println("")
    pretty_table(stdout, data; kw...)
    printstyled("p-values. "; color=:yellow)
    println("𝜒² tests null β=0.")
    println("")


    return nothing
end

"""
`B, BStd, e, R2 = mv_reg(y::Matrix{Float64}, x; constant=true, S=white(), R2adj=false)`

`B[i,:]` stores the coefficients of the `i`-th equation.

`S` can be `preset(x)` for a given `x`, `nw(k)` or `hh(k)` for given number of lags, or still `white()`

`BStd` contains the standard deviation of each coefficient
"""
function mv_reg(y::Matrix{Float64}, x; constant=true, S=white(), R2adj=false)
    T, nY = size(y)
    nX = size(x, 2)
    B = zeros(nY, nX + constant)
    BStd = zeros(nY, nX + constant)
    e = zeros(T, nY)
    R2 = zeros(nY)
    for j = 1:nY
        B[j, :], bCov, e[:, j], R2[j] = reg(y[:, j], x; constant=constant, S=S, R2adj=R2adj)
        BStd[j, :] = sqrt.(diag(bCov) / T)
    end
    return B, BStd, e, R2
end

"""
`pc, λ, V = principal_components(X; remove_mean=false, positive_eigvec=true)`

Option `positive_eigvec` forces most elements of eigenvectors to be positive

`pc[:,i]` stores the `i`-th principal component series

`λ` stores the eigenvalues in increasing order

`V[:,i]` contains `i`-th eigenvector
"""
function principal_components(X; remove_mean=false, positive_eigvec=true)
    remove_mean && (X = demean(X))
    λ, V = eigen(cov(X), sortby=z -> -z)

    if positive_eigvec
        Id = [count(col .< 0) > count(col .>= 0) for col in eachcol(V)]
        V .*= ones(size(V, 1)) * ((.!Id') .+ -1 * (Id'))
    end
    pc = X * V

    return pc, λ, V
end