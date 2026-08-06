using Documenter: Documenter
using FlowInvariantTransfer: FlowInvariantTransfer

# Only FlowInvariantTransfer's own modules are documented here; the shared SpectralBackends /
# ComputationalBackends packages carry their own docs and are linked to (not re-documented), so docs
# needs only Documenter + FlowInvariantTransfer as dependencies.
Documenter.makedocs(;
    modules  = [FlowInvariantTransfer,
        FlowInvariantTransfer.Types, FlowInvariantTransfer.Utils,
        FlowInvariantTransfer.Invariants, FlowInvariantTransfer.Decomposition, FlowInvariantTransfer.ShellBinning,
        FlowInvariantTransfer.Filters, FlowInvariantTransfer.Workspaces, FlowInvariantTransfer.NonlinearTerm,
        FlowInvariantTransfer.SpectralFlux, FlowInvariantTransfer.CoarseGrainingFlux,
        FlowInvariantTransfer.ShellToShellTransfer, FlowInvariantTransfer.BandTransfer,
        FlowInvariantTransfer.TriadicOrthogonalDecomposition, FlowInvariantTransfer.ModeToModeTransfer,
        FlowInvariantTransfer.Compressible, FlowInvariantTransfer.Spherical],
    sitename = "FlowInvariantTransfer.jl",
    authors  = "Jordan Benjamin",
    format   = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical  = "https://jbphyswx.github.io/FlowInvariantTransfer.jl",
        edit_link  = "main",
    ),
    pages = [
        "Home"                 => "index.md",
        "Methods & Theory"     => "methods.md",
        "Architecture"         => "architecture.md",
        "Backends & Extensions" => "backends.md",
        "API Reference"        => "api.md",
    ],
    # `:cross_references` is warned (not fatal) because FIT documents symbols from the shared
    # ComputationalBackends package whose docstrings use bare `@ref`s that cannot resolve under this
    # page scope (a ComputationalBackends-side doc gap, tracked upstream).
    warnonly = [:missing_docs, :docs_block, :cross_references],
)

Documenter.deploydocs(;
    repo      = "github.com/jbphyswx/FlowInvariantTransfer.jl",
    target    = "build",
    branch    = "gh-pages",
    devbranch = "main",
)
