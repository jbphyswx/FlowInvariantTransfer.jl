using Documenter: Documenter
using FlowInvariantTransfer: FlowInvariantTransfer

Documenter.makedocs(;
    modules  = [FlowInvariantTransfer,
        FlowInvariantTransfer.Types, FlowInvariantTransfer.Backends, FlowInvariantTransfer.Utils,
        FlowInvariantTransfer.Invariants, FlowInvariantTransfer.Decomposition, FlowInvariantTransfer.ShellBinning,
        FlowInvariantTransfer.Filters, FlowInvariantTransfer.Workspaces, FlowInvariantTransfer.NonlinearTerm,
        FlowInvariantTransfer.SpectralFlux, FlowInvariantTransfer.CoarseGrainingFlux,
        FlowInvariantTransfer.ShellToShellTransfer, FlowInvariantTransfer.BandTransfer,
        FlowInvariantTransfer.TriadicOrthogonalDecomposition, FlowInvariantTransfer.ScaleToScaleTransfer,
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
    warnonly = [:missing_docs, :docs_block],
)

Documenter.deploydocs(;
    repo      = "github.com/jbphyswx/FlowInvariantTransfer.jl",
    target    = "build",
    branch    = "gh-pages",
    devbranch = "main",
)
