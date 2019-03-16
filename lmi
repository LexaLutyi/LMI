using LinearAlgebra
using SparseArrays

function scalar_m(X, A, B)
    """Scalar product <.,.>ₚ with P = Xₖ"""
    return tr(X * A * X * B)
end


function gramian(X, A)
    """Gramian matrix calculation function for system Aᵢ
    with scalar production scalar_m"""
    Q = zeros(length(A), length(A))
    for i, j in 1:length(A)
        Q[i,j] = scalar_m(X, A[i], A[j])
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
    for i in 1:size(A)[1], j in i:size(A)[1]
        E = spzeros(Number, size(A)...)
        E[i ,j], E[j, i] = 1, 1
        R = transpose(A) * E + E * A
        push!(Q, R)
    end
    return Q
end


A = [[  10  1   -8;1   8   -6;-8 -6    8],
    [ 4 5 -1; 5 -4 3;-1 3 -12],
    [ 4 -1 0;-1 2 -7; 0 -7 8],
    [10 -6 -8;-6 -2 5;-8 5 6],
    [-10 -3 5; -3 10 -7;5 -7 -4],
    [-10 0 -7; 0 -10 3; -7 3 -8]]
X = Matrix{Number}(I, 3, 3)

S = solve_lmi_sum(X, A, 100)

sum(S .* A)

"""
Aᵀ * X + X * A < 0
∑-(Aᵀ * Eᵢⱼ + Eᵢⱼ * A)xᵢⱼ > 0
"""
A = [-2 1; 3 -2]
eigmax(A)
