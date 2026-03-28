# Suppress Qt/Wayland warnings
ENV["QT_QPA_PLATFORM"] = "xcb"
ENV["QT_QUICK_BACKEND"] = "software"
ENV["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"

using Pkg
Pkg.add(["CSV", "DataFrames", "FileIO"])
using CSV, DataFrames, FileIO 
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

function five_point_cubic_interpolation(a, b, step, func, x)
    n = round(Int, (b - a) / step)
    knots = build_knots(a, b, step)
    result = 0.0
    for i in 1:(n+3)
        p1 = knots[i+1]
        p3 = knots[i+2]
        p2 = (p1 + p3)/2
        p5 = knots[i+3]
        p4 = (p5 + p3)/ 2
        w =  [1.0, -8.0, 20.0, -8.0, 1.0] / 6.0
        pts = [p1, p2, p3, p4, p5]
        coeff = sum(w .* func.(pts))
        result += coeff * b_spline(i-1, 3, x, knots)
    end
    return result
end

function add_gaussian_noise(f, noise_level=0.01)
    return x -> f(x) + noise_level * randn()
end

test_funcs = [
    ("x^3 - 2x + 1", x -> x^3 - 2x + 1),
    ("x^4", x -> x^4),
    ("x^2", x -> x^2),
    ("x", x->x),
    ("sin(x)", sin),
    ("exp(x)", exp),
    ("|10x|^0.5", x->abs(10*x)^(0.5)),
    ("rung_1_(1+25x²)", x->1/(1+25x^2)),
    ("sin(1_x)", x->sin(1/(x+10.001)))
]

function run_tests(a, b, step_values)
    results = DataFrame(
        step = Float64[],
        function_name = String[],
        noise = Bool[],
        method = String[],
        max_error = Float64[]
    )

    for step in step_values
        println("\n" * "="^60)
        println("Step size: $step")
        println("="^60)

        for (name, f_orig) in test_funcs
            println("\n--- Testing function: $name ---")

            grid = fine_grid(a, b, step)
            err_var = Float64[]
            err_five = Float64[]
            for x in grid
                exact = f_orig(x)
                approx_var = variation_reducing(a, b, step, f_orig, x)
                approx_five = five_point_cubic_interpolation(a, b, step, f_orig, x)
                push!(err_var, abs(exact - approx_var))
                push!(err_five, abs(exact - approx_five))
            end
            max_var = maximum(err_var)
            max_five = maximum(err_five)
            push!(results, (step, name, false, "variation_reducing", max_var))
            push!(results, (step, name, false, "five_point", max_five))

            plot_approximation(a, b, step, f_orig, f_orig, name, step, false)

            println("  Without noise:")
            println("    Variation reducing max error: $max_var")
            println("    Five-point max error: $max_five")

            f_noisy = add_gaussian_noise(f_orig, 0.01)
            err_var_noisy = Float64[]
            err_five_noisy = Float64[]
            for x in grid
                exact = f_orig(x)           
                approx_var = variation_reducing(a, b, step, f_noisy, x)
                approx_five = five_point_cubic_interpolation(a, b, step, f_noisy, x)
                push!(err_var_noisy, abs(exact - approx_var))
                push!(err_five_noisy, abs(exact - approx_five))
            end
            max_var_noisy = maximum(err_var_noisy)
            max_five_noisy = maximum(err_five_noisy)

            push!(results, (step, name, true, "variation_reducing", max_var_noisy))
            push!(results, (step, name, true, "five_point", max_five_noisy))
            plot_approximation(a, b, step, f_orig, f_noisy, name, step, true)

            println("  With noise (σ=0.01):")
            println("    Variation reducing max error: $max_var_noisy")
            println("    Five-point max error: $max_five_noisy")
        end
    end

    CSV.write("spline_approximation_results.csv", results)
    println("\nResults saved to spline_approximation_results.csv")
end

function plot_approximation(a, b, step, exact_func, approx_func, name, step_val, noisy)
    fine = fine_grid(a, b, step)
    x_vals = collect(fine)
    y_exact = exact_func.(x_vals)
    y_approx = [approx_func(x) for x in x_vals]

    plt = plot(x_vals, y_exact, label="Exact", linewidth=2)
    plot!(plt, x_vals, y_approx, label="Approximation", linestyle=:dash)
    title!(plt, "$name (step = $step_val)")

    folder = noisy ? "plots/noisy" : "plots/wo_noise"
    mkpath(folder)

    filename = joinpath(folder, "$(replace(name, " " => "_"))_step_$step_val.png")
    savefig(plt, filename)
    display(plt)
end

step_values = [0.8, 0.253, 0.08]
run_tests(-10, 10, step_values)