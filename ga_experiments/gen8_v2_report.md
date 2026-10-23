# ga-gen8-v2-layout-emit-an8-and-deoverlap

- **Branch**: `ga/gen8-v2-layout-emit-an8-and-deoverlap`
- **Description**: Gen-8 V2: Strip stale {\an8} on non-overlapping an2 OSTs in layout.py and emit.py, enforce OST single-quoting, and de-overlap same-channel (an2/an8) display events with positive duration guarantee
- **Official Gold Standard**: `20/20 (100.00)`
- **R01..R20 Strict Live Invariant Suite**: `54/54 PASS (27 Flash + 27 Opus, 0 failures)`
- **Unsnapped .000 OST Events**: `Flash 132 -> 0, Opus 133 -> 0`
- **Unnecessary {\an8} (No Dialogue Overlap)**: `Flash 6 -> 0, Opus 9 -> 0 (ost_pos = 100.0%)`
- **Same-Channel Collisions**: `Flash 0 -> 0, Opus 42 -> 0`
- **Jevify 27-Work OST/Timing Benchmark**: `0/27 -> 26/27 clean_and_synchronized (Mean P(zero_defects): 0.031 -> 0.941)`
