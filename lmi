using LinearAlgebra
using SparseArrays


"""Scalar product <.,.>ₚ with P = Xₖ"""
scalar_m(X, A, B) = tr(X * A * X * B)


function gramian(X, A)
    """Gramian matrix calculation function for system Aᵢ
    with scalar production scalar_m"""
    Q = zeros(length(A), length(A))
    for i in 1:length(A)
        for j in 1:length(A)
            Q[i,j] = scalar_m(X, A[i], A[j])
        end
    end
    return Q
end

vect_m(X, A) = map(p->scalar_m(X, X, p), A)

function ortho_project(X, A)
    """Orthogonal projection of matrix Xₖ onto Range(A) = {y | ∃ x: y = Ax}"""
    gramian_A  = gramian(X, A)
    right_part = vect_m(X, A)

    coords = gramian_A \ right_part
    coords = map(p -> isnan(p) ? 0 : p, coords)
    return sum(coords .* A)
end

function gamma(X, A)
    X_ortho = ortho_project(X, A)
    sqrt_matr = inv(sqrt(X))
    ψ = sqrt_matr * (X_ortho - X) * sqrt_matr
    ρ = maximum(abs.(eigvals(ψ)))
    return 1/(1+ρ)
end

function solve_lmi_sum(X, A, max_step)
    X_ortho = zeros(size(X))
    for i in 1:max_step
        X_ortho = ortho_project(X, A)
        if isposdef(X_ortho)
            break
        end
        γ = gamma(X, A)
        X⁻¹ = inv(X)
        X = X⁻¹ - γ * X⁻¹ * (X_ortho - X) *  X⁻¹
        X = inv(X)
    end

    if  isposdef(X_ortho)
        println("Победа вместо обеда")
        sol = gramian(X, A) \ vect_m(X, A)
        println(X_ortho, eigvals(X_ortho))
        return map(p -> isnan(p) ? 0 : p, sol)
    else
        println("Точно не попали")
        return nothing
    end
end

function lmi_normalize(A₀)
    Q = []
    for i in 1:size(A₀)[1], j in i:size(A₀)[1]
        E = spzeros(Number, size(A₀)...)
        E[i ,j], E[j, i] = 1, 1
        R = transpose(A₀) * E + E * A₀
        push!(Q, -R)
    end
    return Q
end
