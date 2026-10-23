# ga-gen8-v3-refeval-strict-session-guard

- **Branch**: `ga/gen8-v3-refeval-strict-session-guard`
- **Description**: Gen-8 V3: Harden refeval.py check_session_invariants (R01..R20) to execute live same-channel collision, ost_pos==100, ost_emp==100, R19 zero unsnapped .000 events, and R20 DP&W SPAM timing checks without cache bypass
- **Official Gold Standard**: `20/20 (100.00)`
- **R01..R20 Strict Live Invariant Suite**: `54/54 PASS (27 Flash + 27 Opus, 0 failures)`
- **Unsnapped .000 OST Events**: `Flash 132 -> 0, Opus 133 -> 0`
- **Unnecessary {\an8} (No Dialogue Overlap)**: `Flash 6 -> 0, Opus 9 -> 0 (ost_pos = 100.0%)`
- **Same-Channel Collisions**: `Flash 0 -> 0, Opus 42 -> 0`
- **Jevify 27-Work OST/Timing Benchmark**: `0/27 -> 26/27 clean_and_synchronized (Mean P(zero_defects): 0.031 -> 0.941)`
