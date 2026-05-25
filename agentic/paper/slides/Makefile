# ============================================================================
# ISCS 2026 slides build + submission
# ----------------------------------------------------------------------------
# Targets:
#   make                    -> both PDFs
#   make slides.pdf         -> clean 16:9 conference version
#   make slides-presenter.pdf
#                           -> wide pages (slide | notes) for pdfpc
#   make present            -> build presenter PDF + launch pdfpc
#   make submission         -> build the ISCS 2026 zip:
#                                39_Alexander_Towell.zip containing
#                                39_Alexander_Towell.pdf + .mp4
#                              (expects a video file at VIDEO_SRC)
#   make watch              -> rebuild both whenever slides.tex changes
#   make clean              -> remove LaTeX auxiliaries
#   make distclean          -> also remove PDFs and the submission zip
# ============================================================================

LATEX     := pdflatex -interaction=nonstopmode -halt-on-error
SRC       := slides.tex
PDF       := slides.pdf
PDF_PRES  := slides-presenter.pdf
PRES_DEF  := "\def\presentermode{} \input{$(SRC)}"

# Submission file naming (per ISCS 2026 email):
#   [SUBMISSION_ID]_[PRESENTER_NAME]_[PRESENTER_SURNAME].zip
SUBMISSION_ID := 39
PRESENTER     := Alexander_Towell
STEM          := $(SUBMISSION_ID)_$(PRESENTER)
SUB_PDF       := $(STEM).pdf
SUB_MP4       := $(STEM).mp4
SUB_ZIP       := $(STEM).zip
VIDEO_SRC     ?= recording/final.mp4

.PHONY: all present watch clean distclean submission

all: $(PDF) $(PDF_PRES)

# Conference build: standard 16:9, no notes panel.
$(PDF): $(SRC)
	$(LATEX) $(SRC)
	$(LATEX) $(SRC)

# Presenter build: pages are slide+notes side by side. pdfpc consumes this.
$(PDF_PRES): $(SRC)
	$(LATEX) -jobname=slides-presenter $(PRES_DEF)
	$(LATEX) -jobname=slides-presenter $(PRES_DEF)

# Launch pdfpc in presenter view on the presenter PDF.
present: $(PDF_PRES)
	pdfpc $(PDF_PRES)

# Build the ISCS 2026 submission zip.
# Requires VIDEO_SRC to point at the final encoded MP4. Defaults to
# recording/final.mp4. The video must be <= 150 MB and <= 12 min.
submission: $(PDF) $(VIDEO_SRC)
	@if [ ! -f $(VIDEO_SRC) ]; then \
	  echo "ERROR: video not found at $(VIDEO_SRC)"; \
	  echo "Set VIDEO_SRC=path/to/final.mp4 or place file at recording/final.mp4"; \
	  exit 1; \
	fi
	@size=$$(stat -c%s $(VIDEO_SRC)); \
	limit=$$((150 * 1024 * 1024)); \
	if [ $$size -gt $$limit ]; then \
	  echo "ERROR: video is $$(($$size / 1024 / 1024)) MB, ISCS limit is 150 MB"; \
	  echo "Re-encode with: see RECORDING_SETUP.md '2-pass encode' section"; \
	  exit 1; \
	fi
	cp $(PDF) $(SUB_PDF)
	cp $(VIDEO_SRC) $(SUB_MP4)
	rm -f $(SUB_ZIP)
	zip $(SUB_ZIP) $(SUB_PDF) $(SUB_MP4)
	@echo
	@echo "Submission package ready: $(SUB_ZIP)"
	@ls -lh $(SUB_PDF) $(SUB_MP4) $(SUB_ZIP)
	@echo
	@echo "Upload at: https://cmt3.research.microsoft.com/ISCS2026/"
	@echo "  Action column -> Upload Supplementary Material -> $(SUB_ZIP)"

# Auto-rebuild on edit (requires entr: sudo apt install entr).
watch:
	@echo $(SRC) | entr -c $(MAKE) all

clean:
	rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb

distclean: clean
	rm -f $(PDF) $(PDF_PRES) $(SUB_PDF) $(SUB_MP4) $(SUB_ZIP)
