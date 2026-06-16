using Documenter: Documenter
using FlowInvariantTransfer: FlowInvariantTransfer
using FlowInvariantTransfer
using FFTW: FFTW

Documenter.makedocs(;
    modules  = [FlowInvariantTransfer],
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
