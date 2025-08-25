# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is an academic research paper repository focused on applying complex network analysis to ChatGPT conversations. It contains LaTeX source files for a research paper and presentation analyzing conversational AI interactions through network science methodologies.

## Project Structure

```
.
├── alex-chatgpt-complex-net.tex      # Main research paper (LaTeX)
├── alex-chatgpt-complex-net.pdf      # Compiled paper
├── alex-pres-complex-networks.tex    # Presentation slides (Beamer)
├── alex-pres-complex-networks.pdf    # Compiled presentation
└── images/                            # 47+ visualizations
    ├── *.png                          # Network visualizations
    ├── *.pdf                          # LaTeX-compatible figures
    ├── *.svg                          # Source vector graphics
    └── export-svg.sh                  # SVG to PDF conversion script
```

## Common Development Commands

### Building Documents

```bash
# Compile the main paper
pdflatex alex-chatgpt-complex-net.tex
bibtex alex-chatgpt-complex-net        # If references change
pdflatex alex-chatgpt-complex-net.tex  # Second pass for references
pdflatex alex-chatgpt-complex-net.tex  # Third pass for final formatting

# Compile the presentation
pdflatex alex-pres-complex-networks.tex

# Convert SVG images to PDF (if needed)
cd images && ./export-svg.sh
```

### LaTeX Compilation Tips

- Use multiple passes when changing citations or cross-references
- Check for missing package errors and install required LaTeX packages
- Beamer presentations may require additional passes for overlays

## Key Components

### Main Paper (`alex-chatgpt-complex-net.tex`)
- Academic paper analyzing 449 ChatGPT conversations as semantic networks
- Uses standard LaTeX article class with academic formatting
- Includes complex mathematical notation and network analysis algorithms
- References external implementation at: https://github.com/queelius/chatgpt-complex-net

### Presentation (`alex-pres-complex-networks.tex`)
- Beamer presentation summarizing research findings
- Contains animated overlays and progressive reveals
- Designed for academic conference presentation

### Images Directory
- Network visualizations showing conversation topology
- Statistical plots (degree distributions, clustering coefficients)
- Community detection results
- Both raster (PNG) and vector (PDF/SVG) formats

## Important Notes

1. **This is a publication repository** - No executable code, focuses on LaTeX documents
2. **External code reference** - Actual implementation at https://github.com/queelius/chatgpt-complex-net
3. **Image dependencies** - Many figures referenced in LaTeX must exist in images/
4. **Citation management** - Paper uses BibTeX for references (embedded in .tex file)

## Research Context

The paper introduces "cognitive MRI" - using network analysis to reveal hidden patterns in AI conversations. Key findings include:
- 15 distinct knowledge communities with 0.75 modularity
- Three types of bridge conversations (evolutionary, integrative, pure)
- Non-standard degree distribution challenging scale-free assumptions

## LaTeX Package Requirements

Essential packages used:
- `graphicx` - Image inclusion
- `tikz` - Network diagrams
- `algorithm2e` - Algorithm presentation
- `booktabs` - Professional tables
- `hyperref` - Clickable references
- `beamer` - Presentation framework (for slides only)