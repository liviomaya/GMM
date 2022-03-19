demean(X::Vector) = X .- mean(X)
demean(X::Matrix) = mapslices(demean, X, dims = 1)
g_apply(f, X, b) = mean(f(X, b), dims = 1)[:]

# √T (b̂ - b) → 𝑁(0, bCov)
# √T g → 𝑁(0, gCov) (gCov non-invertible, use pinv instead)
# X2_stat → Χ²(M-P)
"""
b, g, dg, s, bCov, gCov, X2__stat = gmm_spec(X, f::Function, b0, a; df=forwarddiff(), S=white())

Solve GMM model `a' Ef(X;b) = 0`.
`b0` is the initial guess. 

`df` can be `exact(df)` for a given function `df(x,b)` or `forwarddiff(;step=1e-5)`.

`S` can be `preset(x)` for a given `x`, `nw(k)` or `hh(k)` for given number of lags, or still `white()`.

# Asymptotic Distributions

√T (b̂ - b) → 𝑁(0, bCov)

√T g → 𝑁(0, gCov) (gCov non-invertible, use pinv for inference)

X2_stat → χ²(M-P)

"""
function gmm_spec(X, f::Function, b0, a; df = forwarddiff(), S = white())
    @assert size(a, 1) == length(b0) "Number of linear combinations of moments must match number of parameters"

    P, M = size(a)
    T = size(X, 1)

    # solve aᵀg = 0
    obj(b) = a * g_apply(f, X, b)
    r = nlsolve(obj, b0)
    !converged(r) && error("No solution to aᵀg = 0.")
    b = r.zero

    # calculate g, dg/db and S = T Var(g)
    g_func(x, par) = g_apply(f, x, par)
    g = g_func(X, b)
    DF = df(f)
    dg = mean(DF(X, b), dims = 1) |> x -> reshape(x, (M, P))
    s = S(X, f, b)

    # asymptotic distributions
    bCov = inv(a * dg) * a * s * a' * inv(a * dg)'
    gCov = (I(M) - dg * inv(a * dg) * a) * s * (I(M) - dg * inv(a * dg) * a)'
    #return b, g, 0, 0, 0, 0, 0

    # Χ² test statistic
    X2_stat = T * g' * pinv(gCov) * g

    return b, g, dg, s, bCov, gCov, X2_stat
end

# √T (b̂ - b) → 𝑁(0, bCov)
# √T g → 𝑁(0, gCov) (gCov non-invertible, use pinv instead)
# J_stat → Χ²(M-P)
function gmm_step(X, f::Function, b0, W; df = forwarddiff(), S = white())

    M = size(W, 1)
    P = length(b0)
    T = size(X, 1)

    # minimize gᵀ W g
    obj(b) = g_apply(f, X, b)' * W * g_apply(f, X, b)
    r = optimize(obj, b0, BFGS(), Optim.Options(iterations = 200, show_trace = false, show_every = 10))
    r = optimize(obj, r.minimizer, Newton(), Optim.Options(iterations = 25, show_trace = false))
    !Optim.converged(r) && println("Minimization of gᵀ W g failed")
    b = r.minimizer

    # calculate g, dg/db and S = T Var(g)
    g_func(x, par) = g_apply(f, x, par)
    g = g_func(X, b)
    DF = df(f)
    dg = mean(DF(X, b), dims = 1) |> x -> reshape(x, (M, P))
    s = S(X, f, b)

    # asymptotic distributions
    bCov = inv(dg' * W * dg) * dg' * W * s * W' * dg * inv(dg' * W' * dg)
    gCov = (I(M) - dg * inv(dg' * W * dg) * dg' * W) * s * (I(M) - dg * inv(dg' * W * dg) * dg' * W)'

    # J statistic
    X2_stat = T * g' * pinv(gCov) * g
    # X2_stat = T * g' * pinv(s) * g     if W = S⁻¹

    return b, g, dg, s, bCov, gCov, X2_stat
end

"""
b, g, dg, s, bCov, gCov, X2_stat = gmm(X, f, b0, N::Int64=1; W=collect(I(length(f(X[1,:], b0)))), df=forwarddiff(), S=white())

Solve GMM model `Min Ef(X;b)' W Ef(X;b)`.
`b0` is the initial guess. `N` is the number of iterations on the spectral density matrix. 

`df` can be `exact(df)` for a given function `df(x,b)` or `forwarddiff(;step=1e-5)`.

`S` can be `preset(x)` for a given `x`, `nw(k)` or `hh(k)` for given number of lags, or still `white()`.

# Asymptotic Distributions

√T (b̂ - b) → 𝑁(0, bCov)

√T g → 𝑁(0, gCov) (gCov non-invertible, use pinv for inference)

X2_stat → χ²(M-P) (J test if  W → S⁻¹)

"""
function gmm(X, f, b0, N::Int64 = 1; W = collect(I(size(f(X, b0), 2))), df = forwarddiff(), S = white())
    T = size(X, 1)
    b, g, dg, s, bCov, gCov, X2_stat = gmm_step(X, f, b0, W; df = df, S = S)
    for n = 2:N
        if n < N
            S0 = S # recalculate every step prior to last
        elseif n == N
            S0 = preset(s)
        end
        b, g, dg, s, bCov, gCov, X2_stat = gmm_step(X, f, b, inv(s); df = df, S = S0)
        X2_stat = T * g' * pinv(s) * g
    end
    return b, g, dg, s, bCov, gCov, X2_stat
end