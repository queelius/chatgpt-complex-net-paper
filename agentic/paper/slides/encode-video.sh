#!/usr/bin/env bash
# 2-pass H.264 encode of video-slides.mkv -> final.mp4 (AAC stereo, faststart).
# Optionally speed up via setpts + atempo (pitch preserved).
#
# Usage:
#   ./encode-video.sh                       # 1.0x
#   ./encode-video.sh --speed 1.05          # 1.05x (imperceptible)
#   ./encode-video.sh --speed 1.1           # 1.1x (barely noticeable)
#   ./encode-video.sh -i other.mkv -o other.mp4 --speed 1.05
#
# Speed guidance (voice): 1.05 invisible, 1.10 barely noticeable,
# 1.20 slightly brisk, 1.30+ starts to feel rushed.

set -euo pipefail
cd "$(dirname "$0")"

INPUT="video-slides.mkv"
OUTPUT="final.mp4"
SPEED="1.0"
VBITRATE="1500k"
ABITRATE="192k"

while [ $# -gt 0 ]; do
    case "$1" in
        --speed)     SPEED="$2"; shift 2 ;;
        --speed=*)   SPEED="${1#*=}"; shift ;;
        -i|--input)  INPUT="$2"; shift 2 ;;
        -o|--output) OUTPUT="$2"; shift 2 ;;
        --bitrate)   VBITRATE="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,13p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [ ! -f "$INPUT" ]; then
    echo "ERROR: input $INPUT not found" >&2
    echo "  (run ./make-pres-video.sh first to create video-slides.mkv)"
    exit 1
fi

# Build the filter chain. Skip filters entirely at 1.0x.
if [ "$SPEED" = "1.0" ] || [ "$SPEED" = "1" ]; then
    VFILTER_PASS1=()
    VFILTER_PASS2=()
    echo "Encoding $INPUT -> $OUTPUT (2-pass, 1.0x)..."
else
    VFILTER_PASS1=( -filter_complex "[0:v]setpts=PTS/${SPEED}[v]" -map "[v]" )
    VFILTER_PASS2=( -filter_complex "[0:v]setpts=PTS/${SPEED}[v];[0:a]atempo=${SPEED}[a]" -map "[v]" -map "[a]" )
    echo "Encoding $INPUT -> $OUTPUT (2-pass, ${SPEED}x, pitch preserved)..."
fi

ffmpeg -y -i "$INPUT" \
    "${VFILTER_PASS1[@]}" \
    -c:v libx264 -preset slow -b:v "$VBITRATE" \
    -pass 1 -an -f mp4 /dev/null

ffmpeg -y -i "$INPUT" \
    "${VFILTER_PASS2[@]}" \
    -c:v libx264 -preset slow -b:v "$VBITRATE" \
    -c:a aac -b:a "$ABITRATE" \
    -pass 2 -movflags +faststart "$OUTPUT"

rm -f ffmpeg2pass-0.log ffmpeg2pass-0.log.mbtree

SIZE_MB=$(( $(stat -c%s "$OUTPUT") / 1024 / 1024 ))
echo
echo "Output: $OUTPUT (${SIZE_MB} MB)"
if [ "$SIZE_MB" -gt 150 ]; then
    echo "WARNING: exceeds 150 MB ISCS limit. Try: --bitrate 1300k"
fi
ffprobe -hide_banner -i "$OUTPUT" 2>&1 | grep -E 'Duration|Stream' || true
