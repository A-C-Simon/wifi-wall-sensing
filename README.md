# wifisense — WiFi sensing through walls on a MacBook

Device-free presence and motion sensing using only the laptop's built-in WiFi
and a router. Includes a radar/PPI display, fingerprint-based coarse
localisation, and a scripted validation protocol.

**Read this first.** Everything below was measured on the target machine, not
assumed. The headline finding is that this hardware cannot do what the
"see through walls" demos do, and the measurements say exactly why. The project
is built to extract all of the signal that genuinely exists and to be honest
about the rest.

---

## Measured capability envelope

Host: MacBookAir9,1 (2020, Intel i3-1000NG4), macOS 15.8.1, x86_64.
Radio: Broadcom **BCM4377B3** (`14e4:4488`), FullMAC with closed firmware.

| Channel | Poll rate | Actual information content |
|---|---|---|
| `CWInterface.rssiValue()` | **174–194 Hz** | 1 dB quantised; 7 integer levels spanning ~6 dB; register transitions ~0.2–1 Hz |
| `CWInterface.transmitRate()` | ~1120 Hz | 1 level (static 585–866 Mbps; link has ~33 dB SNR margin) |
| `CWInterface.noiseMeasurement()` | ~180 Hz | static |
| `scanForNetworks()` (all visible APs) | **1 fresh vector / 12.1 s** | 18–32 anchors × RSSI |

**Total information bandwidth ≈ 0.2 Hz.** That single number governs everything.

### What is impossible here, and why

- **CSI (channel state information)** — every "through the wall" paper you will
  read uses per-subcarrier phase/amplitude. This needs Intel 5300, Qualcomm
  Atheros AR9580, or ESP32. BCM4377 is a FullMAC Broadcom part running closed
  firmware on its own ARM core; there is no code path from it to a userspace CSI
  buffer on macOS, with or without root.
- **Monitor mode / raw 802.11** — the `airport` binary
  (`Apple80211.framework/.../Resources/airport`) still exists but is a
  **deprecation stub** on macOS ≥14.4: it prints a warning and exits 0 with no
  output, for every flag. No sniff, no scan, nothing.
- **Per-frame RSSI** — not exposed to userspace on macOS at all. `ioreg` has no
  RSSI key; `netstat -I` has no RSSI column.
- **SSID / BSSID** — redacted unless the *calling binary* holds Location Services
  **and** has a stable code signature. APs are therefore keyed on
  `(channel, width)`, which collapses co-channel APs.
- **Real-time tracking** — at 0.08 Hz for the spatial channel, Nyquist is
  0.04 Hz. A walking human moves at ~1 m/s. Tracking is not a tuning problem;
  the samples do not exist.

The 12 s scan refresh is a **hard global OS limit**. Verified: `includeHidden:
true` does not help, and three concurrent processes each observed exactly the
same 5 sweeps — the limiter is global, not per-process.

---

## Quick start

Use Python 3.10 or newer. Capturing WiFi requires macOS, CoreWLAN and the
Xcode command line tools (`swiftc`). The live displays also require Tk support
in your Python installation. Offline analysis and regression tests can run
without the WiFi interface or a desktop session.

```bash
cd wifi_wall_sensing
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt

# 1. What can this machine actually do? (re-measures, ~70 s)
python3 wifisense.py doctor

# 2. Live radar display (free-running)
python3 wifisense_radar.py --router-bearing 315 --router-range 4.5

# 3. Record a session
python3 wifisense.py capture --duration 300

# 4. Threshold from an EMPTY room (do this first — see below)
python3 wifisense.py baseline --duration 300

# 5. Analyse it
python3 wifisense.py analyze out/capture_*.jsonl
```

`wifisense_live.py` is a second, panel-style dashboard (scrolling traces +
anchor bars) if you prefer traces to a radar.

Run the regression checks with:

```bash
MPLBACKEND=Agg python3 -m unittest discover -s tests -v
```

If your shell exports `PYTHONPATH` from a different Python environment, unset
it before running these commands. Mixing extension modules from different
Python versions can prevent NumPy from importing.

---

## Why the threshold must be measured, not chosen

With 1 dB quantisation and slow drift, a threshold picked from theory is
meaningless. `baseline` records the score distribution while the room is known
to be empty and sets the threshold at `median + k·1.4826·MAD`. Without this
step the detector will happily alarm on an empty room all afternoon.

The baseline now saves its fine-channel center and scale, sweep anchor set,
and sweep reference as well as its threshold. Analysis and live displays use
those saved parameters rather than normalizing each measurement independently.
Regenerate older `out/baseline.json` files with `baseline`; incompatible files
are rejected. The analysis window must match the baseline window. An explicit
live `--threshold` selects descriptive scoring without the saved calibration.

Fine-channel windows use only past samples. Sweep evidence begins at the
completed scan timestamp and expires using that scan's preceding interval.
Missing or stale live samples produce `NO DATA` rather than a clear-room
decision.

---

## Two honest limitations in the localisation

**1. RSSI cannot give distance by itself.** Received power follows
`RSSI(d) = A − 10·n·log10(d)`, which has two unknowns (`A`, `n`) and one
observable. Any single RSSI reading is consistent with a whole family of
distances. Measured on this host, the associated link sits at −55 dBm / 33 dB
SNR, which is consistent with anything from ~3 m to ~8 m depending on
obstructions.

The tool therefore does not guess. `python3 wifisense.py calibrate-distance`
has you stand at measured distances (default 1, 2, 3.5, 5, 7 m) and
least-squares fits `A` and `n`, saving `out/linkmodel.json`. The radar then
places every anchor by range from that fitted model and shows the fitted
`n`, so you can see how much the estimate is worth. Sanity check: fitting a
synthetic survey with `n=3.100` recovers `n=3.078`.

**2. Bearing of the other access points is not observable.** With one receiver
you cannot infer an AP's bearing without knowing its position. The radar
spreads unknown anchors deterministically by channel hash to keep the display
readable. That is a **layout convenience, not a measurement** — do not read
those positions as real. The router's bearing is a real, user-supplied input.

---

---

---

## Result: RSSI motion detection also fails on this hardware

Scripted EMPTY / WALK / STILL / EMPTY, 235 s, scored against the exact schedule
the operator was given:

| phase | median score | p95 | alarms |
|---|---|---|---|
| EMPTY-1 | 0.67 | 2.02 | 0% |
| **WALK** | **0.06** | 1.67 | 0% |
| **STILL** | **1.86** | 2.41 | 0% |
| EMPTY-2 | 0.67 | 1.77 | 0% |

```
WALK  vs EMPTY : -0.83 sigma     walking looks QUIETER than an empty room
STILL vs EMPTY : +1.63 sigma     standing still looks like motion
WALK  vs STILL : -2.46 sigma     backwards
detection      : NOT DETECTED
```

**Walking produced the lowest score of all four phases.** Raw channel
statistics for the same windows:

| phase | RSSI | sd | span | register transitions/s |
|---|---|---|---|---|
| EMPTY-1 | −55.0 | 1.44 | 8.0 dB | 3.85 |
| WALK | −55.7 | 1.23 | 9.0 dB | 4.25 |
| STILL | −54.0 | 0.95 | 6.0 dB | 2.84 |
| EMPTY-2 | −54.8 | 0.99 | 6.4 dB | 2.35 |

The register toggles between 1 dB levels 2.4–4.3 times per second *constantly*,
in every phase, empty or occupied. That ambient switching is larger than
anything the body contributes, so the fused score tracks link instability rather
than motion. No phase reached the threshold.

Note the direction: the link was *strongest* while the operator stood still and
*weakest* while walking. That is consistent with the physics — a stationary body
is a persistent obstruction giving a steady interference offset, whereas a
walking body decorrelates the channel so fast that a 12 s snapshot averages it
away. Even so the effect is buried under ambient variation.

Reproduce: `python3 wifisense.py --iface en0 capture --duration 235`, then
`python3 validate/score.py <file> 3.0 10.0 70 75 45`

---

## Result: RSSI fingerprint localisation does NOT work on this hardware

This was tested, not assumed. Three independent measurements agree.

### 1. Within-session cell classification: 0% accuracy

Leave-one-out over sweep halves, 4 regions 4 m apart, chance = 25%:

```
accuracy : 0.0%  (0/8)      <- every query matched the wrong region
chance   : 25.0%
lift     : 0.00x
```

Reproduce: `python3 wifisense.py cv-test`

### 2. Session drift exceeds the spatial signal

The same four positions were surveyed twice, ~15 minutes apart:

| measurement | value |
|---|---|
| same position, different session | **40.0 dB** (1.4–2.9 dB per anchor) |
| different positions, within a session | **43.5 dB** |
| worst same-position drift | 63.0 dB — exceeds 3 of 6 spatial separations |

A body at the *same* spot moves the fingerprint as much as a body at a
*different* spot. Since every live vector is matched against a map built
earlier, this drift is the noise floor localisation actually lives with.

Reproduce: `python3 wifisense.py repeat-test`

### 3. Cells are not separable from the noise

```
within-cell temporal noise : 13.8 dB
closest region pair        : 16.8 dB  (1.23x noise)   need >= 2x
verdict                    : MARGINAL
```

A finer 3x3 survey at 2 m spacing was also run: 31 of 36 pairs passed at ratio
2, but **all 5 failures were adjacent 2 m neighbours**, and the largest mutually
separable subset spanned 4 x 4 m. Merging those cells into 4 coarse regions
made it *worse* (1.21x) — averaging sub-positions 2 m apart cancels exactly the
position dependence being measured.

Reproduce: `python3 wifisense.py calibrate` then `validate-map`

#---

## Result: occupancy is not detectable either, and the pooled figure lies

The tables below record the original scoring pipeline. Current scores differ
after the causal window and calibration fixes. The occupancy command now
computes uncertainty across A/B pairs, rather than treating thousands of
overlapping fine buckets as independent observations. Pooled and centered AUC
remain descriptive statistics. A centered AUC also removes a genuine shift in
occupancy level, so it cannot establish absence of an occupancy effect by
itself. Fixed A/B order still needs randomized confirmation before attributing
a repeatable difference to occupancy.

Three A/B pairs of 90 s blocks, alternating empty and occupied, 9.3 min total.

| block | condition | median score |
|---|---|---|
| 1 | EMPTY | -0.80 |
| 2 | OCCUPIED | 0.00 |
| 3 | EMPTY | 0.17 |
| 4 | OCCUPIED | 4.71 |
| 5 | EMPTY | **4.85** |
| 6 | OCCUPIED | 2.89 |

Pooling every sample gives **AUC 0.640, p = 0.0005**, which reads as a weak
but real effect. It is not real. The score ramps monotonically across the
capture and the highest block is an *empty* one, so the pooled figure mostly
measures the ramp.

The design alternates, so the ramp can be cancelled two ways:

| analysis | AUC | 95% CI |
|---|---|---|
| pooled, raw | 0.640 | 0.630 to 0.651 |
| paired with the bracketing empty block | 0.699 mean | per pair 0.694 / 0.939 / 0.465 |
| pooled after removing each block's own median | **0.456** | 0.445 to 0.467 |

Once each block's baseline is removed the score carries no information about
occupancy, and is slightly *anti*-correlated. The per-pair AUCs disagree with
each other, so pair 2's 0.939 is noise.

**The general lesson: with a drifting measurement, a pooled test will happily
report p < 0.001 for nothing.** Any experiment on this hardware needs a
blocked or baseline-subtracted design, and the baseline has to be collected in
the same session as the measurement.

Reproduce: `python3 validate/occupancy_test.py --block 90 --reps 3`

## Why the literature agrees with these results

None of the four negative results is a misconfiguration. Searching the
literature for RSSI-only sensing on unmodified commodity hardware:

- **The slowest sampling rate in any peer-reviewed commodity passive people
  counting paper is 10 Hz** (Li et al., *Passive People Counting using Commodity
  WiFi*), and that work uses nine CSI amplitude streams from a 3x3 array, not
  one scalar. This machine delivers 0.083 Hz for the multi-AP channel.
- **802.11bf use-case KPIs** (Ropitault et al., IEEE ComSoc Standards Magazine
  2023) put presence detection at 10 to 100 ms and respiration below 50 ms.
  The 90 s blocks used above are already far slower than the fastest KPI in
  the standard.
- **Indoor RSSI decorrelates in single-digit milliseconds under motion.**
  Coherence time at 1 m/s is about 6 ms at 2.4 GHz and 3 ms at 5 GHz. A 2 to
  4 s walking event sampled every 12.1 s gives roughly a one in twelve chance
  of being observed at all, quantised to a single integer step, on a signal
  whose entire usable span is about 6 dB. That is event-limited, and no
  threshold placement recovers it.
- **1 dB quantisation is a studied limiter.** Luong, Zhou and Cummings,
  *RSS Step Size: 1 dB is not Enough!* (Utah SPAN) show that with 1 dB steps
  "the disruption either may not be measured at all, or the spectral features
  calculated from the quantized RSS may be too noisy", and that breathing
  estimation needs roughly 12 bits. WiRSSI (arXiv:2602.14004) independently
  notes RSSI carries "about 0.5 to 1 dB" integer steps and that "the Doppler
  extracted from RSSI ... is strongly affected by coarse amplitude
  quantization".
- **The sign of an RSSI change is not informative.** Depatla and Mostofi's own
  framework has two competing mechanisms, LOS blockage reducing power and
  multipath scattering increasing it, which is why their methods use
  *distributional* statistics over many events rather than any mean or
  variance over a short window. That is exactly what "walking scores below
  empty" looks like.
- **Radio map drift is documented and large.** *Sensors* 2024 (PMC10903848)
  measures RSSI correlated with indoor relative humidity at r = -0.78 on
  2.4 GHz and r = -0.81 on 5 GHz, with the offset growing over ten months with
  no human movement at all. The 40 dB of drift measured here over 15 minutes
  is the same phenomenon at a shorter timescale.
- **The one published RSSI-only motion result that is both fast and accurate
  is 5.69 s mean latency** (Korany and Mostofi, *Nocturnal Seizure Detection
  Using Off-the-Shelf WiFi*, IEEE IoT-J 2021, arXiv:2103.13556: 93.85%
  detection, PFA 0.0097). Its "off the shelf" means off the shelf NICs in a
  purpose-deployed transmitter and receiver pair, not the NIC already inside
  the laptop.

### MCS and transmit rate as a sensing channel

Worth ruling out explicitly, since it looks promising: `transmitRate()` is
fast to read and reacts to instantaneous SINR. It is a dead end here for two
reasons. No published work uses link rate adaptation as a sensing signal; in
the CSI literature it is consistently treated as a *nuisance* to be removed,
named as a source of amplitude impulse noise in CARM (MobiCom 2015). And at
-55 dBm against -88 dBm noise the link has 33 dB of SNR on 80 MHz 5 GHz, so
rate adaptation is pinned against its ceiling with no headroom. A single
static level is what a saturated link looks like.

### Two unquantified confounds

- **AWDL.** `awdl0` is up and running on this host. Apple's peer to peer radio
  shares the WiFi hardware and is documented to pull the NIC off channel, and
  the 12.1 s scan interval may be partly an artefact of that contention rather
  than a floor. Not tested, because disabling it needs root and altering the
  host's network configuration was out of scope. The measurement in this
  README should be read as an upper bound on the scan rate, not a proven floor.
- **Scan staleness.** A round robin sweep hears access points on different
  channels at different moments, so a vector assembled across channels is not
  internally time coherent, and entries for access points not heard in a given
  round may be carried over from the driver cache. macOS exposes no per
  access point timestamp, so this cannot be corrected from userspace.
  *Computer Communications* 2024 ("Last seen time is critical") makes the same
  point about RSSI indoor localisation generally.

## What would actually work

Ranked by measured evidence rather than by how good the idea sounds:

1. **An ESP32-S2 or S3 running 802.11mc fine timing measurement.** WiFine
   (Nikseresht et al., DCOSS-IoT 2023) reports **95.8% occupancy accuracy on
   single antenna commodity WiFi**, with 8 hours of data and a sampling
   requirement 10x lower than a CSI baseline. This is the best single radio
   result in the area and it needs no CSI at all. CoreWLAN exposes no ranging
   API, so this is not reachable from macOS.
2. **An ESP32-S3 for CSI**, at 20 to 100 Hz. Hernandez et al. (ICC 2021)
   report perfect detection and false alarm rates over 120 s windows.
3. **A second radio you control**, Atheros AR9580 or an Intel 5300 under
   Linux, for full CSI at 100 to 2500 Hz. Every strong RSSI-only result in the
   Depatla and Mostofi lineage is literally this setup.
4. **Accept 0.083 Hz and change the task** to quasi-static "is anyone home"
   over 5 to 30 minute windows, same-session calibration only, with humidity
   as a modelled covariate. Do not expect it to survive re-deployment on
   another day.

The sensing code here ports across to any of these. `features.py` and
`radiomap.py` consume a samples by anchors matrix regardless of whether the
values came from CSI, FTM or RSSI. What cannot port is the acquisition layer,
and that is the whole problem.

### What the system does about it

`validate-map` writes a verdict into the map, and **the radar refuses to
display positions when the verdict is not `usable`**. Motion contacts are still
drawn, but labelled "position unresolved", and the header says
`LOCALISATION OFF`. Without that gate the radar would have painted a confident,
wrong position on every scan — the worst possible failure mode for a sensor.

### Why this happens

The environment drifts faster than the target moves the measurement. In this
room, over 15 minutes, 28 anchors became 27, channel populations shifted, and
per-anchor values moved 1.4–2.9 dB at a fixed position. The body's contribution
is a small perturbation on top of that. On paper a 2 m displacement should
change an anchor by a few dB; measured against a 1 dB-quantised, heavily damped
register with ~2 dB of unexplained session drift, it is not recoverable.

This is not a tuning failure. It is the same information-bandwidth wall as
everything else on this machine: 1 dB quantisation, one receiver, and a scan
refresh of 12 s.

---

## Localisation approach

Single receiver means no triangulation, so position comes from *spatial*
differences in the multi-AP vector: each visible AP is a passive anchor at a
different physical location, so the RSSI vector is a fingerprint whose value
depends on where the body is.

```bash
# survey a grid: stand at each cell, ~42 s per cell (3 sweeps minimum)
python3 wifisense.py calibrate --cols 3 --rows 3 --extent 2.0

# attribute a later capture to cells
python3 wifisense.py localize out/capture_xxx.jsonl
python3 wifisense.py heatmap  out/capture_xxx.jsonl
```

Distance is mean absolute RSSI difference in dB across measured shared
anchors. Spatial variation between cells is the localization signal, so it
is no longer used as a proxy for temporal noise to down-weight anchors.
Every candidate cell uses the same observed dimensions. A query needs at
least five shared anchors covering at least half the map's anchor set;
otherwise it returns no position. Missing readings are not copied between
cells or replaced with a synthetic live fingerprint.

The radar, `localize`, and `heatmap` require a map that passes `validate-map`.
Coarsening a map or recording a new repeatability result invalidates its old
verdict. Repeatability compares matched positions on the same measured anchor
set in both surveys. Map matching prepares its matrix once for repeated
queries, and separation checks reduce cell pairs in bounded memory.

---

## Validation protocol

`validate/walk_test.py` runs a scripted EMPTY → WALK → STILL → EMPTY sequence
and scores detection latency, false-alarm rate, and how many empty-room sigmas
separate walking from standing still. `validate/score.py` scores a capture
against a known schedule.

The radar's `--mode walktest` runs the same protocol with on-screen prompts so
you can follow along without reading a terminal.

**A caveat learned the hard way:** phase boundaries must match what the
operator was actually told. An earlier run mislabelled 14 s of walking as
"empty", which inverted the result. When you run this, take the schedule from
the on-screen countdown rather than from a chat message.

---

## Bugs worth knowing about (all found and fixed here)

- `set_theta_zero_location('N')` + `set_theta_direction(-1)` does **not** give a
  clockwise-from-north frame on current matplotlib — 0° lands at the *bottom*.
  The radar is therefore Cartesian with explicit `x = r·sin(b)`, `y = r·cos(b)`.
- A detector with a fixed 60 s detrend half-life accumulates AGC drift over a
  long capture; an *empty* phase scored higher than the walk phase. Half-life
  now scales with capture length.
- Frame-to-frame multi-AP change is physically **stationary**, so normalising
  it by its own median beats a z-score, which drifts and eventually makes a
  quiet stretch look anomalous.
- Leading NaNs in the anchor matrix must be back-filled as well as
  forward-filled; one unfilled gap poisons the median of the whole sweep-change
  series and silently zeroes the score.
- `String(format:)` with a `%@` and a raw `UnsafePointer<CChar>` is undefined
  behaviour and segfaults inside Foundation. Use string interpolation.
- A VPN `utun` interface owning the default route makes `connect()`+`send()` to
  the LAN gateway fail with `EADDRNOTAVAIL`; ICMP works. Traffic generation
  uses `ping` for this reason.

---

## If you want the real thing

The blocker is hardware, not software. The highest-leverage purchase:

| Purchase | Cost | Unlocks |
|---|---|---|
| **2× ESP32-S3** | ~$20–40 | Full CSI at 20–100 Hz, 56–192 effective subcarriers. [ESP32-CSI-Tool](https://github.com/espressif/esp-csi) is actively maintained. This moves you to the tier nearly all modern sensing papers occupy: pose, vitals, through-wall HAR. |
| 1× ESP32-S3 + router as the other end | ~$15 | 1Tx-1Rx CSI link, cheapest possible upgrade from dead to alive |
| Atheros AR9580 + PCIe adapter | ~$30–60 | Atheros CSI Tool; also runs OpenWrt. Only for 3×3 MIMO. |

The sensing software here ports across largely unchanged — `features.py` and
`radiomap.py` consume a `[samples × anchors]` matrix regardless of whether the
values came from CSI or RSSI. Replace `wifisense_core` with a CSI tap and the
DSP, calibration and validation all carry over.

### Literature this design follows

- Depatla & Mostofi, *Crowd Counting Through Walls Using WiFi*, IEEE PerCom 2018
  — RSSI-only, Tx/Rx both outside the walls; ≤2-person error in 100% of 44
  experiments, up to 20 people. The strongest evidence that RSSI-only
  through-wall sensing is real.
- Depatla, Muralidharan & Mostofi, *Occupancy Estimation Using Only WiFi Power
  Measurements*, IEEE JSAC 33(7), 2015 — 98% presence / 77% exact count.
- Mager, Lundrigan & Patwari, *Fingerprint-based device-free localization in
  changing environments*, IEEE JSAC 33(11), 2015 — the fingerprint approach
  used here.
- Abdelnasser, Youssef & Harras, *WiGest*, IEEE INFOCOM 2015 — RSSI-only
  gesture recognition, 96%, using multiple overheard APs as anchors.
- Adib & Katabi, *See Through Walls with Wi-Fi!*, SIGCOMM 2013 — the actual
  "through walls" result: USRP N210 software-defined radios, 3-antenna arrays.
  Note the hardware gap.

---

## Layout

```
native/wifisense_core.swift   CoreWLAN capture daemon (both channels -> JSONL)
sensing/io.py                 session parsing, sweep-burst de-duplication
sensing/features.py           detrend, envelope, robust z, sweep statistics
sensing/detect.py             fused motion decision + empty-room calibration
sensing/radiomap.py           fingerprint map, anchor weighting, localisation
sensing/linkmodel.py          log-distance fit for RSSI <-> range
sensing/viz.py                matplotlib analysis plots
wifisense.py                  CLI: doctor/capture/baseline/analyze/calibrate/...
wifisense_radar.py            radar PPI display
wifisense_live.py             panel dashboard
validate/walk_test.py         scripted protocol
validate/score.py             scoring against a known schedule
probe/                        the feasibility experiments (exp1-exp7)
```

`probe/` is kept deliberately: those experiments produced the capability table
above, and re-running them is how you check the numbers on a different machine.

Two earlier drafts are not in the repository. One drove traffic from a raw UDP
socket, which cannot work here because a utun interface owns the default route
and `connect()` to the LAN gateway fails; ICMP replaced it in exp3. The other
measured the same accessor timings as exp6 and was superseded when it crashed.
Neither contributed a finding that the kept experiments do not cover.

## Commit layout

One commit per idea, roughly in the order the ideas were needed. The probe
commits establish what the hardware can do, the sensing commits implement
against those limits, and the validation commits try to prove the result. The
last validation commits report failures, which is why the README leads with
limits rather than features.
