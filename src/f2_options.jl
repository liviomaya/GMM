#= 

df can be any of the following:
    exact(df) 
    forwarddiff()

S can be any of the following:
    preset(S)
    nw(k)
    hh(k) 
    white() =#


function exact(df)
    function f_gen(f)
        return df
    end
    return f_gen
end

function forwarddiff(; step=1e-5)
    function f_gen(f)

        function df(x, b)
            # evaluate function at point b
            ff = f(x, b)

            # get relevant sizes
            T, M = size(ff)
            P = length(b)

            # calculate directional derivative dual
            dff = zeros(T, M, P)
            for p = 1:P
                # b change
                db = zeros(P)
                db[p] = step

                # evaluate f at b + db
                ff_fwd = f(x, b + db)
                dff[:, :, p] .= (ff_fwd .- ff) ./ step
            end
            return dff
        end

        return df
    end
    return f_gen
end

function S_estimator(k::Int64, w::Function)
    function S_gen(X, f, b)
        M = size(f(X, b), 2)
        T = size(X, 1)
        F = f(X, b) |> demean
        S = zeros(M, M)
        for j in range(-k, k, step=1)
            wei = w(j, k)
            (wei == 0) && continue

            ja = abs(j)
            if j > 0
                v_r = F[ja+1:end, :] # enters multiplication as row vector
                v_c = F[1:end-ja, :]
            elseif j < 0
                v_r = F[1:end-ja, :] # enters multiplication as row vector
                v_c = F[ja+1:end, :]
            elseif j == 0
                v_r = F
                v_c = F
            end
            av = (v_r' * v_c) / T
            S += wei * av
        end
        # S = (S .+ S') / 2
        return S
    end
    return S_gen
end

nw(k::Int64) = S_estimator(k, (j, k) -> (k - abs(j)) / k)
hh(k::Int64) = S_estimator(k, (j, k) -> 1)
white() = S_estimator(2, (j, k) -> (j == 0) ? 1 : 0)

function preset(S)
    S_gen(X, f, b) = S
    return S_gen
end