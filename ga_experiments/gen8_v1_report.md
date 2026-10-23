# ga-gen8-v1-ost-timing-and-shot-snap

- **Branch**: `ga/gen8-v1-ost-timing-and-shot-snap`
- **Description**: Gen-8 V1: Fix wall-clock-second (:SS) chunk-offset addition drift in narrative.py/translate.py (pinning DP&W SPAM to 01:16:34.465..01:16:38.469) and snap .000 integer-second OST timestamps to 23.976fps frame boundaries
- **Official Gold Standard**: `20/20 (100.00)`
- **R01..R20 Strict Live Invariant Suite**: `54/54 PASS (27 Flash + 27 Opus, 0 failures)`
- **Unsnapped .000 OST Events**: `Flash 132 -> 0, Opus 133 -> 0`
- **Unnecessary {\an8} (No Dialogue Overlap)**: `Flash 6 -> 0, Opus 9 -> 0 (ost_pos = 100.0%)`
- **Same-Channel Collisions**: `Flash 0 -> 0, Opus 42 -> 0`
- **Jevify 27-Work OST/Timing Benchmark**: `0/27 -> 26/27 clean_and_synchronized (Mean P(zero_defects): 0.031 -> 0.941)`
