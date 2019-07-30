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

    coords = qr(gramian_A, Val(true)) \ right_part
    if ~isapprox(
            norm(gramian_A * coords - right_part),
            0;
            atol=length(gramian_A)*eps(Float64)
        )
        error("Ошибка слишком велика, возможно, система не разрешима")
    end
    return sum(coords .* A)
end

function gamma(X, A)
    X_ortho = ortho_project(X, A)
    sqrt_matr = inv(sqrt(X))
    ψ = sqrt_matr * (X_ortho - X) * sqrt_matr
    ρ = maximum(abs.(eigvals(ψ)))
    return 1/(1+ρ)
end

function solve_standart_lmi(A, max_step)
    X = Matrix{Float64}(I, size(A[1])...)
    X_ortho = zeros(size(A))
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
        gramian_A  = gramian(X, A)
        right_part = vect_m(X, A)
        coords = qr(gramian_A, Val(true)) \ right_part
        if ~isapprox(
                norm(gramian_A * coords - right_part),
                0;
                atol=length(gramian_A)*eps(Float64)
            )
            error("Ошибка слишком велика, возможно, система не разрешима")
        end
        return coords
    else
        println("Точно не попали")
        return nothing
    end
end

function remove_constant_matrix(∑₁, A₀)
    """
        ∑₁Aᵢxᵢ + A₀ > 0 ⇒ ∑₀Aᵢyᵢ > 0
    """
    ∑₀ = map(p -> [ p    zeros(size(p, 2), 1);
                    zeros(1, size(p, 1))    0],
            ∑₁)
    push!(∑₀, [ A₀    zeros(size(A₀, 2), 1);
                zeros(1, size(A₀, 1))    1])
    return ∑₀
end

function lmi_normalize(A)
    "AᵀX + XA > 0 ⇒ ∑Aᵢxᵢ > 0"
    Q = []
    for i in 1:size(A)[1], j in i:size(A)[1]
        E = zeros(Number, size(A)...)
        E[i ,j], E[j, i] = 1, 1
        R = transpose(A) * E + E * A
        push!(Q, R)
    end
    return Q
end

function lmi_vector(C)
    "cᵀyᵀ + yc > 0 ⇒ ∑qᵢyᵢ > 0"
    Q = []
    m, n = reverse(size(C))
    for i in 1:m, j in 1:n
        E = zeros(Number, (m,n)...)
        E[i,j] = 1
        push!(Q, transpose(C)*transpose(E) + E*C)
    end
    return Q
end

function get_real_x(y)
    x = y[1:length(y)-1] / y[length(y)]
end

function get_matrix_X(x, n)
    Q = zeros(n, n)
    i = 1
    j = 1
    for p in x
        E = zeros(n, n)
        E[i ,j], E[j, i] = 1, 1
        Q +=  p * E
        if j == n
            i += 1
            j = i
        else
            j += 1
        end
    end
    return Q
end

function make_standart_lmi_from_system_of_lmi(A...)
    Z = []
    for i in 1:length(A[1])
        q = map(p->sparse(p[i]), A)
        push!(Z, blockdiag(q...))
    end
    return Z
end
