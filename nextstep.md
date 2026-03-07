# Next Step: Improved Cre+ Neuron Classification

## Background

In the Allen Institute Visual Behavior Neuropixels (VBN) dataset, two transgenic mouse lines express
channelrhodopsin-2 (ChR2) in specific interneuron populations:

- **Sst-IRES-Cre** — ChR2 expressed in SST (somatostatin) interneurons
- **Vip-IRES-Cre** — ChR2 expressed in VIP (vasoactive intestinal peptide) interneurons

Because ChR2 makes neurons fire in response to blue light, we can identify which recorded neurons
are actually SST or VIP cells by shining a laser on the brain and seeing which units respond.
This technique is called **optotagging**. Units that respond reliably to the laser are called
**Cre-positive (Cre+)**, and those that don't are **Cre-negative (Cre-)**.

Importantly, the wild-type (`wt/wt`) mice have no ChR2, so optotagging does not apply to them.
All units recorded from WT sessions are treated as unlabeled (effectively all Cre-negative), and
serve as a control population for comparison.

---

## What the Optotagging Data Looks Like

Each session's `optotagging_table` contains rows for laser pulse trials, each with:
- `start_time` — when the pulse began
- `duration` — how long the pulse lasted (short pulses ~10ms are used for classification)
- `level` — laser power level

From these trials, we build a 3D array:
```
opto_array shape: (n_neurons, n_time_bins, n_trials)
```

Each value is the spike rate (Hz) for a given neuron at a given time bin in a given trial.
Time bins are typically 1 ms wide, spanning from -50 ms to +50 ms around laser onset.

We focus classification on the **shortest, highest-power pulses** (`duration == min`, `level == max`)
because these give the cleanest, most time-locked responses.

---

## Current Approach (from the Reference Notebook)

The reference notebook classifies a neuron as Cre+ if it satisfies **two criteria**, both computed
from the **mean response across trials**:

```python
# 1. Compute mean opto response across selected trials
mean_opto_response = np.nanmean(opto_array[:, :, sel_trials], axis=2)
# shape: (n_neurons, n_time_bins)

# 2. Baseline window: -10 ms to -2 ms before laser onset
baseline_time_idx = (time_array >= -0.010) & (time_array < -0.002)
baseline_rate = np.mean(mean_opto_response[:, baseline_time_idx], axis=1)

# 3. Evoked window: 1 ms to 9 ms after laser onset
evoked_rate_idx = (time_array >= 0.001) & (time_array < 0.009)
evoked_rate = np.mean(mean_opto_response[:, evoked_rate_idx], axis=1)

# 4. Classification thresholds (parameters to tune)
increase_in_FR = 5     # evoked must be >5x the baseline
min_evoked_rate = 50   # evoked must be at least 50 Hz

# 5. Classify
cre_pos_idx = (evoked_rate > min_evoked_rate) & ((evoked_rate / (baseline_rate + 1)) > increase_in_FR)
```

> **Note on the +1:** The `baseline_rate + 1` in the denominator prevents a divide-by-zero error
> for neurons with zero baseline firing. This was a bug in the original single-session notebook
> (`baseline_rate + 1` was incorrectly placed outside the denominator) and has been fixed.

### Problem with this approach

Averaging across trials produces a single mean number per neuron, which **hides trial-to-trial
variability**. A neuron that fires 500 Hz on 5% of trials and nothing on the other 95% has the
same mean evoked rate as a neuron that reliably fires 25 Hz on every trial. Only the second neuron
is truly being driven by the laser. The mean alone cannot distinguish these cases.

---

## Improved Classification: Three Criteria

We add two additional criteria on top of the original two, designed to capture response
**reliability** and **stability**.

### Criterion 1 & 2 (unchanged): Mean Rate Thresholds

Same as above — evoked rate must exceed `min_evoked_rate`, and must be `increase_in_FR` times
greater than baseline. These parameters should be tuned by visual inspection of heatmaps
(see Parameter Tuning section below).

---

### Criterion 3 (new): Trial-by-Trial Reliability

**The idea:** A genuinely ChR2-expressing neuron should fire on most laser trials, not just
occasionally. We compute the fraction of trials on which the neuron fires at least one spike in
the evoked window (1–9 ms). This is called the **hit rate** or **reliability**.

```python
evoked_window_idx = (time_array >= 0.001) & (time_array < 0.009)

# Sum spikes in evoked window per trial for each neuron
# opto_array[:, evoked_window_idx, :] has shape (n_neurons, n_evoked_bins, n_trials)
# Summing over time axis gives shape (n_neurons, n_trials)
evoked_spikes_per_trial = opto_array[:, evoked_window_idx, :][:, :, sel_trials].sum(axis=1)

# Fraction of trials with at least 1 spike in the evoked window
reliability = (evoked_spikes_per_trial > 0).mean(axis=1)  # shape: (n_neurons,)

min_reliability = 0.5  # neuron must fire on at least 50% of trials
```

**What this catches:** A neuron that only sporadically responds to the laser (unreliable, likely
noise or incidental activation) will have low reliability and be excluded, even if its mean evoked
rate looks high due to a few very large spikes.

**Parameter to tune:** `min_reliability` (reasonable range: 0.3 to 0.7)

---

### Criterion 4 (new): Split-Half Consistency

**The idea:** If a neuron is truly driven by the laser, it shouldn't matter which subset of laser
trials you use — it will be classified as Cre+ regardless. We test this by:

1. Splitting the selected opto trials into two halves (first half vs second half)
2. Running the mean-rate classification independently on each half
3. Keeping only neurons that pass in **both** halves

This is essentially a cross-validation check on the optotagging label.

```python
trial_indices = np.where(sel_trials)[0]
mid = len(trial_indices) // 2
half1 = trial_indices[:mid]
half2 = trial_indices[mid:]

def classify_trials(opto_array, trial_idx, time_array, min_evoked_rate, increase_in_FR):
    """Classify neurons as Cre+ using only the specified trial indices."""
    mean_resp = np.nanmean(opto_array[:, :, trial_idx], axis=2)

    baseline_idx = (time_array >= -0.010) & (time_array < -0.002)
    evoked_idx   = (time_array >= 0.001)  & (time_array < 0.009)

    base   = np.mean(mean_resp[:, baseline_idx], axis=1)
    evoked = np.mean(mean_resp[:, evoked_idx],   axis=1)

    return (evoked > min_evoked_rate) & ((evoked / (base + 1)) > increase_in_FR)

pos_half1 = classify_trials(opto_array, half1, time_array, min_evoked_rate, increase_in_FR)
pos_half2 = classify_trials(opto_array, half2, time_array, min_evoked_rate, increase_in_FR)

# A neuron must be Cre+ in both halves to count
cre_pos_consistent = pos_half1 & pos_half2
```

**What this catches:** A neuron that only passes classification due to a lucky cluster of spikes
early in the session (session drift, transient activation, or electrode movement artifacts) will
fail in the other half. This is a strong guard against false positives.

**No new parameter needed** — this reuses the existing `min_evoked_rate` and `increase_in_FR`.

---

### Combined Classification

Putting all criteria together:

```python
cre_pos_idx = (
    cre_pos_consistent     &   # passes mean-rate thresholds in both trial halves
    (reliability > min_reliability)  # fires on enough trials (hit rate)
)
```

---

## Parameter Tuning Strategy

### Parameters to sweep

| Parameter         | Role                                   | Suggested range |
|-------------------|----------------------------------------|-----------------|
| `increase_in_FR`  | Fold-change evoked vs baseline         | 3, 5, 10        |
| `min_evoked_rate` | Minimum absolute evoked rate (Hz)      | 20, 50, 100     |
| `min_reliability` | Minimum fraction of trials with spikes | 0.3, 0.5, 0.7   |

### How to evaluate "works best"

There is no ground truth label, so "best" is evaluated by:

1. **Heatmap separation** — When you plot the optotagging heatmap for Cre+ cells only, do all
   rows show a clear, sharp response starting at 0–1 ms? Are there any rows that look like noise?
   For Cre- cells, is the heatmap flat with no visible laser-driven response?

2. **Scatter plot separation** — On the baseline-rate vs evoked-rate scatter plot, are the
   classified Cre+ neurons clearly above the threshold lines, with no borderline cases?

3. **Downstream functional difference** — After classifying, do Cre+ and Cre- neurons behave
   differently in response to novel vs familiar images? SST and VIP interneurons are known to
   modulate visual responses, so a valid optotagging classification should produce populations
   with distinguishable functional properties.

4. **Cell count plausibility** — SST and VIP interneurons make up roughly 30% and 15% of
   cortical inhibitory neurons respectively, which is a small fraction of all recorded units.
   If your parameter set tags 80% of cells as Cre+, something is wrong. If it tags 0%, your
   threshold is too strict.

### Workflow

1. Open `notebooks/single_session_analysis.ipynb`
2. Pick one representative Sst session and one representative Vip session
3. Try different parameter combinations for each, evaluating using the heatmap + scatter plot
4. Choose one parameter set per genotype that gives the cleanest separation
5. Save those parameters into config YAML files (one per genotype)
6. Run `scripts/run_all_sessions.py` with those configs to process all sessions in batch

**Important:** Do NOT tune parameters separately per brain region. One parameter set per genotype
is scientifically principled — the optotagging laser targets the whole cortex, not a specific
region. Tuning per-region would be overfitting.

---

## Note on Image Sets and Optotagging

The optotagging trials (laser pulses from `opto_table`) happen at a separate time in the session
from the visual task (image presentations). They have nothing to do with image sets G, H, or
shared images. The Cre+/Cre- classification is therefore completely image-set-independent by
construction.

After classification is done, you can (and should) separately examine whether Cre+ cells respond
differently to novel vs familiar images — but that analysis uses the visual task trials, not the
opto trials, and does not affect the classification itself.

---

## Summary of Changes from Original Notebook

| What changed | Why |
|---|---|
| Fixed `baseline_rate + 1` placement (now in denominator) | Prevents divide-by-zero and matches the mathematical intent: evoked / (baseline + 1) |
| Changed `*` to `&` for combining conditions | `*` is multiplication and has wrong operator precedence; `&` is the correct element-wise boolean AND |
| Added trial-by-trial reliability criterion | Mean alone can't detect unreliable responders; reliability captures whether the neuron fires consistently across trials |
| Added split-half consistency check | Cross-validates the classification against session drift, transient artifacts, and sampling noise |
