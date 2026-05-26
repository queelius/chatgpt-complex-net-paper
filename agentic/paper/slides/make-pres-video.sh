#!/usr/bin/env bash
# Stitch slide*.mkv recordings into one MKV (lossless), then optionally
# 2-pass encode to final.mp4 for the ISCS 2026 submission.
#
# Usage:
#   ./make-pres-video.sh                       # stitch only
#   ./make-pres-video.sh --encode              # stitch + encode (1.0x speed)
#   ./make-pres-video.sh --encode --speed 1.1  # stitch + encode at 1.1x speed
#                                              # (atempo preserves pitch)
#
# Inputs: every slide*.mkv in this directory, taken in natural-sort order
# (slide1.mkv, slide2.mkv, ..., slide10.mkv). slides.txt is regenerated
# from that list each run.
#
# Speed range: atempo accepts 0.5-2.0. For voice, 1.1 is imperceptible,
# 1.2 sounds slightly brisk, 1.3+ starts to feel rushed.

set -euo pipefail
cd "$(dirname "$0")"

STITCHED="video-slides.mkv"
FINAL_MP4="final.mp4"
VBITRATE="1500k"
ABITRATE="192k"

# --- argument parsing ---
ENCODE=0
SPEED="1.0"
while [ $# -gt 0 ]; do
    case "$1" in
        --encode)     ENCODE=1; shift ;;
        --speed)      SPEED="$2"; shift 2 ;;
        --speed=*)    SPEED="${1#*=}"; shift ;;
        -h|--help)
            sed -n '2,15p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

# --- collect inputs in natural order ---
shopt -s nullglob
MKVS=( $(ls slide*.mkv 2>/dev/null | sort -V) )
shopt -u nullglob

if [ ${#MKVS[@]} -eq 0 ]; then
    echo "ERROR: no slide*.mkv files in $(pwd)" >&2
    exit 1
fi

echo "Stitching ${#MKVS[@]} file(s):"
printf '  %s\n' "${MKVS[@]}"
echo

# --- regenerate the concat list (overwrites slides.txt) ---
: > slides.txt
for f in "${MKVS[@]}"; do
    printf "file '%s'\n" "$f" >> slides.txt
done

# --- step 1: lossless concat ---
ffmpeg -y -f concat -safe 0 -i slides.txt -c copy "$STITCHED"

echo
echo "Stitched: $STITCHED ($(du -h "$STITCHED" | cut -f1))"
ffprobe -hide_banner -i "$STITCHED" 2>&1 | grep -E 'Duration|Stream' || true
echo

# --- step 2: optional 2-pass encode to MP4 ---
if [ "$ENCODE" -eq 0 ]; then
    echo "To encode the submission MP4 next:"
    echo "  ./make-pres-video.sh --encode              # 1.0x"
    echo "  ./make-pres-video.sh --encode --speed 1.1  # 1.1x (pitch preserved)"
    exit 0
fi

# Build the filter chain. Skip filters entirely at 1.0x to avoid
# a needless re-encode roundtrip on already-clean frames.
if [ "$SPEED" = "1.0" ] || [ "$SPEED" = "1" ]; then
    VFILTER_PASS1=()
    VFILTER_PASS2=()
    echo "Encoding to $FINAL_MP4 (2-pass, ~5 min, 1.0x)..."
else
    VFILTER_PASS1=( -filter_complex "[0:v]setpts=PTS/${SPEED}[v]" -map "[v]" )
    VFILTER_PASS2=( -filter_complex "[0:v]setpts=PTS/${SPEED}[v];[0:a]atempo=${SPEED}[a]" -map "[v]" -map "[a]" )
    echo "Encoding to $FINAL_MP4 (2-pass, ~5 min, ${SPEED}x speed, pitch preserved)..."
fi

ffmpeg -y -i "$STITCHED" \
    "${VFILTER_PASS1[@]}" \
    -c:v libx264 -preset slow -b:v "$VBITRATE" \
    -pass 1 -an -f mp4 /dev/null

ffmpeg -y -i "$STITCHED" \
    "${VFILTER_PASS2[@]}" \
    -c:v libx264 -preset slow -b:v "$VBITRATE" \
    -c:a aac -b:a "$ABITRATE" \
    -pass 2 -movflags +faststart "$FINAL_MP4"

rm -f ffmpeg2pass-0.log ffmpeg2pass-0.log.mbtree

SIZE_MB=$(( $(stat -c%s "$FINAL_MP4") / 1024 / 1024 ))
echo
echo "Final MP4: $FINAL_MP4 (${SIZE_MB} MB)"
if [ "$SIZE_MB" -gt 150 ]; then
    echo "WARNING: exceeds 150 MB ISCS limit. Drop VBITRATE to 1300k and re-run."
fi
ffprobe -hide_banner -i "$FINAL_MP4" 2>&1 | grep -E 'Duration|Stream' || true
