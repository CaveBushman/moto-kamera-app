# AI rollout policy

MotoCam is a broadcast-critical app. Stability wins over AI features.

## Stages

1. `ai.type: disabled`
   - No detector.
   - No AI worker.
   - Manual, tap-to-select tracking, gimbal, video, GPS and telemetry only.

2. `ai.type: simulated` / `ai.type: dot`
   - Software-only simulated detections.
   - No HailoRT.
   - No HEF model load.
   - Used to test AI ASSIST, FULL AI, UI labels, telemetry and gimbal response.
   - `dot` is the bench-test marker mode used while tuning the gimbal loop.

3. `ai.type: hailo_canary`
   - Real Hailo runtime and HEF are tried at low rate.
   - Dot AI remains the fallback if the runtime/model is missing, too slow,
     or repeatedly errors.
   - This is the default staged rollout mode before race use.

4. `ai.type: hailo`
   - Real Hailo runtime and HEF.
   - Enable only after the app is stable with `hailo_canary`.

## Startup rule

AI must not be part of the critical boot path. The main UI, video, gimbal
control and telemetry start first. AI starts only after `ai.startup_delay_s`.

Current default:

```yaml
ai:
  type: hailo_canary
  startup_delay_s: 5.0
  max_fps: 2.0
  max_input_width: 320
```

If the app becomes unstable, switch back to:

```yaml
ai:
  type: disabled
```
