# Asymptotics Analytics - New Features Summary

## Overview
Added comprehensive data loading and main analysis functions to `asymptotics_analytics.jl` for automated processing of "standard-lazy" and "grads-lazy" experiment runs.

## New Functions Added

### 1. `extract_dimension_from_path(path::String)::Int`
Extracts the dimension parameter from directory paths like `MultidimProblem{ABProblem}3`.
- Input: File or directory path
- Output: Dimension as integer (0 if extraction fails)

### 2. `load_experiment_data(data_dir, methods, expected_runs)`
Older version of data loading (kept for compatibility). Not recommended.
- Loads TVmetric files from directory structure
- Returns: `Dict[dim => Dict[method => [scores_run_list]]]`

### 3. `load_experiment_data_by_run(data_dir; methods, expected_runs, rename_methods)` ⭐
Main data loading function. Properly handles multiple runs per method.

**Key Features:**
- Automatically discovers problem directories and runs
- Groups data by dimension and method
- Optional automatic renaming: `"standard-lazy"` → `"plain"` and `"grads-lazy"` → `"grad"`
- Handles parsing of JLD2 files safely with error reporting
- Validates expected number of runs per method-dimension combination

**Arguments:**
```julia
load_experiment_data_by_run(
    data_dir::String,
    methods::Vector{String} = ["standard-lazy", "grads-lazy"],
    expected_runs::Int = 20,
    rename_methods::Bool = true
)
```

**Returns:**
```
Dict[dim => Dict[method => [scores_vector_run1, scores_vector_run2, ...]]]
```
where each `scores_vector` is the convergence curve for one run.

### 4. `main_analysis(data_dir; plot_output_dir, r_c, ν, verbose)` ⭐⭐
All-in-one orchestration function. Recommended for most use cases.

**Features:**
- Loads experiment data automatically
- Computes power law fits for all dimensions and methods
- Prints summary tables and theory comparisons
- Generates comparison plots (requires Plots.jl)
- Returns analysis object for further inspection

**Arguments:**
```julia
main_analysis(
    data_dir::String,
    plot_output_dir::String = "plots",
    r_c::Float64 = 2.5,        # Relative cost (gradient evaluation cost)
    ν::Float64 = 2.5,          # Matérn smoothness parameter
    verbose::Bool = true       # Print progress
)
```

**Returns:** `MultiRunAnalysis` object with:
- `dimensions::Vector{Int}` - List of dimensions analyzed
- `results::Dict` - Structure: `dim => {method => [PowerLawFit...]}`

## Data Structure

Expected directory layout:
```
data/
├── MultidimProblem{ABProblem}1/
│   ├── standard-lazy_1_TVmetric.jld2
│   ├── standard-lazy_2_TVmetric.jld2
│   ├── ...
│   ├── grads-lazy_1_TVmetric.jld2
│   ├── grads-lazy_2_TVmetric.jld2
│   └── ...
├── MultidimProblem{ABProblem}2/
│   ├── ...
│
└── [other problem types]
```

Each `TVmetric.jld2` file should contain a `"score"` key with a vector of scores (convergence curve).

## Usage Examples

### Quick Start (Recommended)
```julia
include("asymptotics_analytics.jl")

# Run complete analysis
analysis = main_analysis("data"; verbose=true)

# Access results
dims = analysis.dimensions
fit = analysis.results[2]["plain"][1]  # First run, dimension 2, plain method
println("Slope β = $(fit.β)")
```

### Manual Control
```julia
# Load data with custom options
data = load_experiment_data_by_run(
    "data";
    methods = ["standard-lazy", "grads-lazy"],
    expected_runs = 20,
    rename_methods = true
)

# Analyze
analysis = analyze_multiple_runs(data; verbose=true)

# Print summaries
summary_table(analysis)
theory_comparison_table(analysis; r_c=2.5, ν=2.5)
```

### Detailed Per-Dimension Analysis
```julia
analysis = main_analysis("data")

for d in analysis.dimensions
    fits_plain = analysis.results[d]["plain"]
    fits_grad = analysis.results[d]["grad"]
    
    β_plain = mean([f.β for f in fits_plain])
    β_grad = mean([f.β for f in fits_grad])
    
    println("Dimension $d: slope ratio = $(β_grad / β_plain)")
end
```

## Example Script

An example script is provided in `example_asymptotics_analysis.jl` with three functions:
1. `example_basic_analysis()` - Quick end-to-end analysis
2. `example_detailed_analysis()` - Manual steps with custom output
3. `example_bootstrap_analysis()` - Uncertainty estimation on synthetic data

Run with:
```julia
julia src/example_asymptotics_analysis.jl
```

## Integration with Existing Code

The new functions work seamlessly with existing functions:
- `fit_powerlaw()` - Fits power law to a single score vector
- `analyze_multiple_runs()` - Processes data dictionary into fits
- `summary_table()` - Pretty-prints results
- `theory_comparison_table()` - Compares to theory predictions
- `plot_convergence_loglog()` - Visualizes convergence curves (requires Plots.jl)
- `plot_slope_comparison()` - Compares slopes across dimensions (requires Plots.jl)

## Error Handling

The loading functions include several safety features:
- **Missing files**: Skipped with warning messages
- **Invalid dimensions**: Directories without numeric suffix are skipped
- **Read errors**: JLD2 files that fail to load are reported with error info
- **Run count mismatch**: Warning if expected_runs differs from loaded runs
- **Empty data**: Error if no data found at all

## Dependencies

**Required:**
- JLD2, Glob (added to imports)
- StatsBase, Statistics, LinearAlgebra, Distributions
- GLM, StatsModels

**Optional:**
- Plots.jl (for visualization - gracefully skipped if unavailable)

## Notes

- Data are loaded in order by run index (e.g., _1, _2, ..., _20)
- Methods are automatically renamed for compatibility with existing analysis functions
- All printed output includes clear section headers for readability
- Verbose mode provides detailed progress information
