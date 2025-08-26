#!/bin/bash
set -euo pipefail
shopt -s nullglob

INKSCAPE_FLAGS="--export-type=pdf --export-latex --export-dpi=300"

for svg in *.svg; do
  base="${svg%.svg}"
  echo "Converting $svg → ${base}.pdf + ${base}.pdf_tex"
  inkscape "$svg" $INKSCAPE_FLAGS -o "${base}.pdf"
done
