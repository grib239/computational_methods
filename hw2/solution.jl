using QuadGK, LinearAlgebra, Printf

function B(i, p, x, t_array)
    j = i + 1 
    if p == 0
        return (t_array[j] <= x < t_array[j+1]) ? 1.0 : 0.0
    else
        res = 0.0
        denom1 = t_array[j+p] - t_array[j]
        if denom1 != 0
            res += ((x - t_array[j]) / denom1) * B(i, p-1, x, t_array)
        end
        denom2 = t_array[j+p+1] - t_array[j+1]
        if denom2 != 0
            res += ((t_array[j+p+1] - x) / denom2) * B(i+1, p-1, x, t_array)
        end
        return res
    end
end

function do_array_for_approx(a, b, h)
    n = round(Int, (b - a) / h)
    return collect(range(a - 3h, b + 3h, length = n + 7))
end

function three_point_functional_coeffs(f, knots)
    p = 3
    num_coeffs = length(knots) - p - 1
    coeffs = zeros(num_coeffs)
    for i in 1:num_coeffs
        p1 = knots[i+1]
        p3 = knots[i+3]
        p2 = (p1 + p3) / 2

        if i == 1
            coeffs[i] = f(p1)
        elseif i == num_coeffs
            coeffs[i] = f(p3)
        else
            coeffs[i] = -0.5 * f(p1) + 2.0 * f(p2) - 0.5 * f(p3)
        end
    end
    return coeffs
end

function spline_colloc_FR2(K, f, a, b, n)
    p = 3
    h = (b - a) / n
    knots = do_array_for_approx(a, b, h)
    num_coeffs = length(knots) - p - 1
    basis = [t -> B(i-1, p, t, knots) for i in 1:num_coeffs]
    mu = three_point_functional_coeffs(f, knots)
    colloc_pts = [(knots[i+1] + knots[i+2] + knots[i+3]) / 3.0 for i in 1:num_coeffs]
    A = zeros(num_coeffs, num_coeffs)
    for i in 1:num_coeffs
        xi = colloc_pts[i]
        for j in 1:num_coeffs
            integrand(t) = K(xi, t) * basis[j](t)
            integral_val = quadgk(integrand, a, b, rtol=1e-8)[1]
            A[i, j] = basis[j](xi) - integral_val
        end
    end
    c = A \ mu
    y_h(t) = sum(c[i] * basis[i](t) for i in 1:num_coeffs)
    return y_h
end


function test_fredholm2()
    println("1. Fredholm")
    test_cases = [
        ("Ex 1", 
         (x,t) -> x*t, 
         x -> x, 
         x -> 1.5*x, 
         0.0, 1.0),

        ("Ex 2", 
         (x,t) -> 1 - 3*x*t, 
         x -> 1, 
         x -> (8 - 6*x)/3, 
         0.0, 1.0),

        ("Ex 3", 
         (x,t) -> min(x,t), 
         x -> 1.0, 
         x -> cos(x), 
         0.0, π)
    ]
    n = 15 
    for (name, kernel, f, exact, a, b) in test_cases
        approx = spline_colloc_FR2(kernel, f, a, b, n)
        grid = range(a, b, length = n*10)
        max_err = maximum(abs(exact(x) - approx(x)) for x in grid)
        println("$name")
        println("Error: $(@sprintf("%.2e", max_err))\n")
    end
end

function solve_uryson(K, f, a, b, n; tol=1e-8, maxiter=30)
    p = 3
    h = (b - a) / n
    knots = do_array_for_approx(a, b, h)
    num_coeffs = length(knots) - p - 1
    basis = [t -> B(i-1, p, t, knots) for i in 1:num_coeffs]
    colloc_pts = [(knots[i+1] + knots[i+2] + knots[i+3]) / 3.0 for i in 1:num_coeffs]
    A = zeros(num_coeffs, num_coeffs)
    for i in 1:num_coeffs
        xi = colloc_pts[i]
        for j in 1:num_coeffs
            A[i, j] = basis[j](xi)
        end
    end
    rhs0 = [f(xi) for xi in colloc_pts]
    c = A \ rhs0
    for iter in 1:maxiter
        y_vals = A * c
        integral_vals = zeros(num_coeffs)
        for i in 1:num_coeffs
            xi = colloc_pts[i]
            integrand(t) = K(xi, t, sum(c[j] * basis[j](t) for j in 1:num_coeffs))
            integral_vals[i] = quadgk(integrand, a, b, rtol=1e-8)[1]
        end
        rhs_new = [f(colloc_pts[i]) + integral_vals[i] for i in 1:num_coeffs]
        c_new = A \ rhs_new
        if norm(c_new - c, Inf) < tol
            c = c_new
            break
        end
        c = c_new
    end
    y_h(t) = sum(c[i] * basis[i](t) for i in 1:num_coeffs)
    return y_h
end

function test_uryson()
    println("\n2. urson equation")
    a, b = 0.0, 1.0
    n = 10
    exact(x) = x
    K(x, t, y) = (x + t) * y^2
    f(x) = x - (x/3 + 1/4)
    approx = solve_uryson(K, f, a, b, n; tol=1e-10, maxiter=20)
    grid = range(a, b, length = 200)
    max_err = maximum(abs(exact(x) - approx(x)) for x in grid)

    println("Eror: $(@sprintf("%.2e", max_err))")
end
test_fredholm2()
test_uryson()
