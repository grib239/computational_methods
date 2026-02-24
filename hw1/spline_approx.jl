using Base.Threads
using Printf
using Plots


function build_knots(a, b, step)
    n = round(Int, (b - a) / step)
    return range(a - 3step, b + 3step, length=n+7)
end

function fine_grid(a, b, step)
    return range(a, b, step=step/10)
end

function b_spline(i, degree, x, knots)
    idx = i + 1
    if degree == 0
        return knots[idx] ≤ x < knots[idx+1] ? 1.0 : 0.0
    else
        val = 0.0
        d1 = knots[idx+degree] - knots[idx]
        if d1 ≠ 0
            val += ((x - knots[idx]) / d1) * b_spline(i, degree-1, x, knots)
        end
        d2 = knots[idx+degree+1] - knots[idx+1]
        if d2 ≠ 0
            val += ((knots[idx+degree+1] - x) / d2) * b_spline(i+1, degree-1, x, knots)
        end
        return val
    end
end

function variation_reducing(a, b, step, func, x)
    n = round(Int, (b - a) / step)
    knots = build_knots(a, b, step)
    result = 0.0
    for i in 1:(n+3)
        center = knots[i + 2]
        coeff = func(center)
        result += coeff * b_spline(i-1, 3, x, knots)
    end
    return result
end

function three_point_scheme(a, b, step, func, x)
    n = round(Int, (b - a) / step)
    knots = build_knots(a, b, step)
    result = 0.0
    for i in 1:(n+3)
        p1 = knots[i+1]
        p2 = knots[i+2]
        p3 = knots[i+3]
        w =  [-1.0, 8.0, -1.0] / 6.0
        pts = [p1, p2, p3]
        valid_idx = findall(p -> try func(p); true catch e; false end, pts)
        coeff = sum(w[valid_idx] .* func.(pts[valid_idx]))
        result += coeff * b_spline(i-1, 3, x, knots)
    end
    return result
end

function add_gaussian_noise(f, noise_level=0.01)
    return x -> f(x) + noise_level * randn()
end

function plot_approximation(a, b, step, func, name)
    fine = fine_grid(a, b, step)
    x_vals = collect(fine)
    y_exact = func.(x_vals)
    y_var = [variation_reducing(a, b, step, func, x) for x in x_vals]
    y_three = [three_point_scheme(a, b, step, func, x) for x in x_vals]

    plt = plot(x_vals, y_exact, label="Точная", linewidth=2)
    plot!(plt, x_vals, y_var, label="Вариационная", linestyle=:dash)
    plot!(plt, x_vals, y_three, label="Трёхточечная", linestyle=:dot)
    title!(plt, "Аппроксимация: $name")
    xlabel!("x")
    ylabel!("f(x)")
    savefig(plt, "$(replace(name, " " => "_")).png")
    display(plt)
end

test_funcs = [
    ("x", x->x),
    ("sin(x)", sin),
    ("exp(x)", exp),
    ("|10x|^0.5", x->abs(10*x)^(0.5)),
    ("rung_1_(1+25x²)", x->1/(1+25x^2)),
    ("sin(1_x)", x->sin(1/(x+10.001)))
]

function run_tests(a, b, step)
    for (name, f) in test_funcs
        #grid = fine_grid(a, b, step)
        #err_var = Float64[]
        #err_three = Float64[]
        #for x in grid
        #    exact = f(x)
        #    approx_var = variation_reducing(a, b, step, f, x)
        #    approx_three = three_point_scheme(a, b, step, f, x)
        #    push!(err_var, abs(exact - approx_var))
        #    push!(err_three, abs(exact - approx_three))
        #end
        #println(name, 
        #    "\n  Вариационная: ", maximum(err_var),
        #    "\n  Трёхточечная: ", maximum(err_three))
        #plot_approximation(a, b, step, f, name)
        #
        name = "noisy_" * name
        f = add_gaussian_noise(f)
        grid = fine_grid(a, b, step)
        err_var = Float64[]
        err_three = Float64[]
        for x in grid
            exact = f(x)
            approx_var = variation_reducing(a, b, step, f, x)
            approx_three = three_point_scheme(a, b, step, f, x)
            push!(err_var, abs(exact - approx_var))
            push!(err_three, abs(exact - approx_three))
        end
        println(name, 
            "\n  Вариационная: ", maximum(err_var),
            "\n  Трёхточечная: ", maximum(err_three))
        plot_approximation(a, b, step, f, name)
    end
end

run_tests(-10, 10, 0.01)