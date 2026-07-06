# AI rollout policy

MotoCam is a broadcast-critical app. Stability wins over AI features.

## Stages

1. `ai.type: disabled`
   - No detector.
   - No AI worker.
   - Manual, tap-to-select tracking, gimbal, video, GPS and telemetry only.

2. `ai.type: simulated`
   - Software-only simulated detections.
   - No HailoRT.
   - No HEF model load.
   - Used to test AI ASSIST, FULL AI, UI labels, telemetry and gimbal response.

3. `ai.type: hailo`
   - Real Hailo runtime and HEF.
   - Enable only after the app is stable with simulated AI.

## Startup rule

AI must not be part of the critical boot path. The main UI, video, gimbal
control and telemetry start first. AI starts only after `ai.startup_delay_s`.

Current default:

```yaml
ai:
  type: simulated
  startup_delay_s: 5.0
```

If the app becomes unstable, switch back to:

```yaml
ai:
  type: disabled
```
