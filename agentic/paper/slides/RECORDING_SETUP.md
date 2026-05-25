# Recording Setup, ISCS 2026 Remote Presentation

Everything needed to record and package the ISCS 2026 submission.
Tooling: **pdfpc** (slide delivery and notes), **OBS Studio** (capture and
encode), **ffmpeg** (compression to hit the 150 MB cap).

## Conference rules that drive the setup

These come from the program-chairs' email. Treat them as hard constraints.

1. **Slot length**: 12 minutes total. 3 additional minutes are for setup,
   intro, and Q&A, not part of the speaker video.
2. **Format**: MP4 video, 150 MB file size cap.
3. **Speaker view is compulsory**. Your face must be visible.
4. **Speaker window must sit in the top-right corner of the screen** and
   must not cover slides.
5. **First slide must include**: title, author names, affiliations, and
   the ISCS 2026 conference logo. Already wired up in `slides.tex`
   (logo at bottom-right of slide 1 so the headshot does not obscure it).
6. **Submission deliverable**: a single zip containing the PDF and the
   MP4, named `39_Alexander_Towell.zip`.

The `make submission` target enforces the file size and naming.

---

## 1. Before you hit record

These matter more than any OBS setting.

### Audio (top priority)

The slides carry the visuals; audio carries you.

- Use a **USB mic** if at all possible. Even a $40 Samson Q2U beats the
  laptop mic by a wide margin. Position 6 to 10 inches from your mouth,
  slightly off-axis so plosives don't pop.
- Record in a **carpeted room** or one with soft furnishings. Bare-wall
  rooms produce echo that's hard to remove later.
- Close windows. Turn off HVAC if you can stand it for 15 minutes. Stop
  the fridge compressor by unplugging briefly (set a reminder to plug
  back in).
- Phone on silent, not vibrate. Vibrate is louder on a desk than ring.

### Video and headshot

- **Camera height**: lens at eye level. Stack books under the laptop. The
  looking-down-at-laptop angle is the single most amateur thing in
  recorded talks.
- **Light from in front, not behind**. A window in front of you
  (daylight), or a desk lamp bounced off a wall in front. Avoid windows
  behind you because they silhouette your face.
- **Background**: plain wall, bookshelf, or one plant. Nothing moving
  (TV off, fan off-camera).
- **Framing**: eyes 1/3 from the top of the headshot crop. Leave a sliver
  of headroom; don't center your face vertically.

### Run a 30-second test

Hit Start Recording in OBS, talk for 30 seconds, stop. Open the file in
`mpv` or VLC. Listen on headphones. Are you peaking? Hissing? Echoing?
Fix those before doing a real take.

---

## 2. OBS scene configuration

One-time setup. Re-runnable via **Tools > Auto-Configuration Wizard**.

### Auto-config wizard

Tools > Auto-Configuration Wizard > **Optimize for recording** >
**1920x1080 at 30 fps**.

### Output settings (Settings > Output)

Set Output Mode to Advanced, then under the Recording tab:

- **Recording Path**: `~/Videos/iscs2026/` (create the directory first).
- **Recording Format**: **MKV** (safer than MP4 if OBS crashes). Final
  conversion to MP4 happens in step 4.
- **Recording Encoder**: x264 software encoder (most compatible).
- **Rate Control**: CRF.
- **CRF**: 18 (visually lossless). The big compression happens later,
  in the 2-pass MP4 encode. Record high-quality master, compress once.
- **Keyframe Interval**: 2 sec.
- **CPU Preset**: veryfast.
- **Profile**: high.
- **Tune**: none.

### Audio settings (Settings > Audio)

- Sample rate: 48 kHz.
- Channels: Stereo.
- Mic/Aux Audio: your USB mic. Test with the **Audio Mixer** panel by
  speaking normally; watch the meter peak around **-12 dB**, no higher
  than -6 dB.

Right-click the mic source, choose **Filters**, then add three:

1. **Noise Suppression** (RNNoise, free). Kills room hum.
2. **Noise Gate**. Threshold around -40 dB, attenuation -30 dB. Mutes
   when you're not talking.
3. **Compressor**. Ratio 4:1, threshold -18 dB. Evens out loud and quiet
   moments.

### Scene: "Talk"

Add three sources. Order controls layering; top of the list draws on top.

1. **Headshot** (top of the list, drawn on top).
   - Source: Video Capture Device (V4L2), set to `/dev/video0`.
   - Capture Resolution 1280x720.
   - **Position the headshot box in the TOP-RIGHT corner** of the canvas.
     ISCS conference rule, not optional.
   - **Recommended size and position for a 1920x1080 canvas**:
     **240x135 box at position (1650, 25)**. This places the headshot
     inside the slide's title-bar zone (which is empty on the right side
     of every slide). Verified against all 11 talk slides; does not
     overlap any content.
   - In OBS: select the Headshot source, then Edit Transform...
       - Position X = 1650, Position Y = 25
       - Bounding Box Type = Scale to inner bounds
       - Bounding Box Size = 240 x 135
   - If you want a larger headshot (more face presence), 280x158 also
     fits. Avoid going larger than 320x180 because that begins to cover
     content on slides 7 and 9 (the right column starts ~y=200).
   - Optional 2 px white outline (Filters > Stroke if you have the
     StreamFX plugin) to separate the headshot from the slide.

2. **Slides** (middle layer).
   - Source: Screen Capture (PipeWire). On selection a portal dialog
     pops up; pick the pdfpc *slide* window (not the presenter window).
   - Resize to fill the canvas.

3. **Mic** (audio only, no visual).
   - Source: Audio Input Capture (PulseAudio), your mic.

Pin the layout: lock the Slides and Headshot sources once positioned so
you can't accidentally drag them.

### Verify the layout before recording

Open the slides PDF in pdfpc and click through every slide while watching
the OBS preview. Confirm the headshot in the top-right does not cover any
content (title text, axis labels, table values, etc.). The deck was laid
out with a clear top-right zone for exactly this reason.

---

## 3. Recording workflow

### Layout for a one-screen session

You don't need to see the slide window during recording. Arrange:

- **OBS preview** on the left half of your screen.
- **pdfpc presenter window** on the right half (slide, next slide, notes,
  timer).
- pdfpc's slide window can be **off-screen, behind other windows, or
  minimised**. OBS Screen Capture grabs its contents regardless.

### Step-by-step recording

1. Open a terminal in this directory.
2. Run `make present`. This builds `slides-presenter.pdf` and launches
   pdfpc.
3. pdfpc opens two windows. Position them as above.
4. In OBS, click **Start Recording**.
5. **Wait 3 seconds before speaking** (gives editing headroom).
6. Press `g` in pdfpc to start its own timer. Cross-check with OBS.
7. Deliver the talk. Read from the notes panel; glance at the lens for
   eye contact periodically.
8. At the end, **wait 3 seconds after your last word** before clicking
   Stop Recording in OBS.

### pdfpc keyboard shortcuts

* `space`, `Right Arrow`, `Down Arrow`, `Page Down`: next slide.
* `Left Arrow`, `Up Arrow`, `Page Up`: previous slide.
* `b`: blank to black (useful if you need to cough off-camera).
* `g`: start or pause the timer.
* `r`: reset the timer.
* `Home`: jump to the first slide.
* `End`: jump to the last slide.
* number then `Enter`: jump to slide N.
* `Esc` or `q`: quit.
* `f`: toggle fullscreen on the slide window.
* `o`: overview (all slides as thumbnails).

### Multi-take strategy

You will not nail this in one take. Plan for 3.

- **Take 1**: a full run to find your pacing. Watch it back with a
  notepad. Note where you stumbled or got vague.
- **Take 2**: a clean run. If you flub one slide, *keep going*. Don't
  restart from the top.
- **Take 3 and beyond**: surgical re-records of specific slides. Slate
  each retake by saying the slide number out loud ("Slide six, take
  two"). Makes splicing painless.

---

## 4. Post-processing with ffmpeg

OBS recordings land in `~/Videos/iscs2026/` by default.

### Why a 2-pass encode is mandatory

Total budget: 150 MB for up to 12 min.

- 150 MB = 150 x 1024 x 1024 x 8 bits / 720 sec = roughly 1.75 Mbps
  total bitrate.
- Reserve 192 kbps for audio (AAC stereo, clean voice). That leaves
  roughly 1.55 Mbps for video.
- At 1080p30 with screen-content slides, 1.55 Mbps is tight but workable
  if you 2-pass encode. A single-pass CRF encode will overshoot or
  undershoot unpredictably.

### Trim the lead and tail silence first (lossless)

```bash
cd ~/Videos/iscs2026
ffmpeg -i take2.mkv -ss 00:00:03 -to 00:12:00 -c copy take2-trimmed.mkv
```

`-ss` and `-to` are start and end; the `-c copy` is instant.

### 2-pass encode to hit the 150 MB target

```bash
TARGET_VBITRATE=1500k        # video bitrate
ABITRATE=192k                # audio bitrate

ffmpeg -y -i take2-trimmed.mkv \
  -c:v libx264 -preset slow -b:v $TARGET_VBITRATE \
  -pass 1 -an -f mp4 /dev/null && \
ffmpeg -i take2-trimmed.mkv \
  -c:v libx264 -preset slow -b:v $TARGET_VBITRATE \
  -pass 2 -c:a aac -b:a $ABITRATE \
  -movflags +faststart \
  final.mp4
```

Notes:

* `-preset slow` trades encoding time for quality. For a 12 min talk it
  takes a few minutes on a laptop. Worth it.
* `-movflags +faststart` puts the MP4 metadata at the start so the file
  streams (the conference platform may stream it during preview).
* If the result still exceeds 150 MB, lower `TARGET_VBITRATE` to 1300k
  and re-run. If it's well under (say 120 MB), bump up to 1700k for
  better quality.

### Check the result

```bash
ls -lh final.mp4
ffprobe -hide_banner final.mp4 2>&1 | grep -E 'Duration|Stream'
```

You want to see:

* `Duration: 00:11:xx` (under 12:00).
* `Stream #0: Video: h264 (High), 1920x1080, 30 fps`.
* `Stream #1: Audio: aac, 48000 Hz, stereo`.
* File size under 150 MB.

### Stitch retakes (if needed)

Create `concat.txt`:

```
file 'take2-part1.mkv'
file 'slide6-retake.mkv'
file 'take2-part2.mkv'
```

Then:

```bash
ffmpeg -f concat -safe 0 -i concat.txt -c copy stitched.mkv
```

Re-run the 2-pass encode on `stitched.mkv` instead of `take2-trimmed.mkv`.

---

## 5. Building the submission zip

`make submission` produces `39_Alexander_Towell.zip` containing the slide
PDF and the video MP4. It verifies the video file size against the 150 MB
limit before zipping.

```bash
cd ~/github/papers/cognitive-mri-ai-conversations/agentic/paper/slides
mkdir -p recording
cp ~/Videos/iscs2026/final.mp4 recording/final.mp4

make submission
```

To override the video location:

```bash
make submission VIDEO_SRC=/some/other/path/final.mp4
```

The output is `39_Alexander_Towell.zip` in this directory, ready to
upload.

---

## 6. Final-submission checklist

Before you upload to CMT:

- [ ] Slide PDF: file `39_Alexander_Towell.pdf` (the renamed conference
      PDF, slide 1 has title + authors + affiliations + ISCS logo).
- [ ] Video MP4: file `39_Alexander_Towell.mp4`, duration under 12:00,
      file size under 150 MB.
- [ ] Headshot visible in **top-right** corner of every video frame.
- [ ] Audio: AAC stereo 48 kHz, no clipping, no echo.
- [ ] Zip file: `39_Alexander_Towell.zip` containing exactly those two
      files.
- [ ] Watch the whole MP4 on headphones, in mpv or VLC, end-to-end. Last
      sanity check before upload.

Inspect the final zip:

```bash
unzip -l 39_Alexander_Towell.zip
```

Expected output:

```
Archive:  39_Alexander_Towell.zip
  Length      Date    Time    Name
---------  ---------- -----   ----
  ~1.1 MB  yyyy-mm-dd hh:mm   39_Alexander_Towell.pdf
   <150MB  yyyy-mm-dd hh:mm   39_Alexander_Towell.mp4
```

---

## 7. If something goes wrong

* **OBS won't pick up the mic.** PulseAudio device locked by another app.
  Close Zoom, Slack, Discord, then restart OBS.
* **Slides window not captureable on Wayland.** Permission portal denied.
  Re-add the Screen Capture source and re-grant when the dialog appears.
* **Webcam shows black.** Another app holding `/dev/video0`. Close
  Cheese, Zoom, and any browser tabs using the webcam.
* **Audio out of sync with video.** OBS dropped frames. Check OBS stats
  panel (View > Stats). Reduce encoder preset and close background apps.
* **pdfpc opens but no notes panel visible.** Opened `slides.pdf` (clean
  conference version) instead of `slides-presenter.pdf`. Use
  `make present` or `pdfpc -w both -n right slides-presenter.pdf`.
* **Final MP4 is over 150 MB.** Drop video bitrate to 1300k or 1200k and
  re-run the 2-pass encode. If still over, drop to 720p:
  `ffmpeg -i in.mkv -vf scale=1280:720 -c:v libx264 ...`.
* **MP4 won't play in QuickTime/Windows player.** Re-mux with
  `ffmpeg -i final.mp4 -c copy -movflags +faststart final-streamable.mp4`.

---

## 8. Day-of-recording sequence

```bash
# 1. Build the presenter PDF.
cd ~/github/papers/cognitive-mri-ai-conversations/agentic/paper/slides
make slides-presenter.pdf

# 2. Open pdfpc.
pdfpc -w both -n right slides-presenter.pdf &

# 3. Open OBS in another terminal (or from the launcher).
obs &

# 4. Confirm OBS layout: slides fill canvas, headshot in top-right,
#    audio meter peaking around -12 dB while you talk.

# 5. Run a 30-second test recording. Watch it back. Adjust if needed.

# 6. Real takes. Slate retakes verbally.

# 7. Trim and 2-pass encode to final.mp4.

# 8. Package and verify.
cp ~/Videos/iscs2026/final.mp4 recording/final.mp4
make submission

# 9. Upload 39_Alexander_Towell.zip to CMT.
```

Good luck.
