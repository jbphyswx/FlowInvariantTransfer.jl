using Documenter: Documenter
using FlowEnergyTransfer: FlowEnergyTransfer
using FFTW: FFTW

Documenter.makedocs(;
    modules  = [FlowEnergyTransfer],
    sitename = "FlowEnergyTransfer.jl",
    authors  = "Jordan Benjamin",
    format   = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical  = "https://jbphyswx.github.io/FlowEnergyTransfer.jl",
        edit_link  = "main",
    ),
    pages = [
        "Home"             => "index.md",
        "Methods & Theory" => "methods.md",
        "API Reference"    => "api.md",
    ],
    warnonly = [:missing_docs, :docs_block],
)

Documenter.deploydocs(;
    repo      = "github.com/jbphyswx/FlowEnergyTransfer.jl",
    target    = "build",
    branch    = "gh-pages",
    devbranch = "main",
)
