Filtering stage (raw -> processed -> filtered -> train/eval)

- Expected review actions format:
  sound_id,action,reason
- Current policy: action == 'exclude' removes sound_id from training/eval.
- This module is not yet wired into trainers; it is a planned pipeline stage.
