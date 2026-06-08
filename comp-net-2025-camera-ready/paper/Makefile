# Makefile for Cognitive MRI of AI Conversations: Analyzing AI Interactions...

TEX = paper.tex
STEM = $(basename $(TEX))
HTMLDIR = html_paper

.PHONY: all pdf html clean

all: pdf

pdf: $(STEM).pdf
html: $(HTMLDIR)/index.html

$(STEM).pdf: $(TEX)
	pdflatex -interaction=nonstopmode $(TEX)
	@if grep -q '\\bibliography' $(TEX) || grep -q '\\addbibresource' $(TEX); then \
		bibtex $(STEM) 2>/dev/null || true; \
	fi
	pdflatex -interaction=nonstopmode $(TEX)
	pdflatex -interaction=nonstopmode $(TEX)

$(HTMLDIR)/index.html: $(TEX)
	@mkdir -p $(HTMLDIR)
	tex2html $(TEX) -o $(HTMLDIR)

clean:
	rm -f *.aux *.log *.out *.toc *.bbl *.blg *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz
	rm -rf $(HTMLDIR)
