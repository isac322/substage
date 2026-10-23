# ga-gen6-v1-snapshot-and-sdh-structure

- **Branch**: `ga/gen6-v1-snapshot-and-sdh-structure`
- **Description**: Gen-6 V1: Fix narrative.py materialize_snapshots speaker/addressee union (0/189 -> 162/189 scenes with VoiceCards/Personas), subs.py extract_accent_sdh_hint, and refeval.py R15..R18 dialect/accent invariants
- **Official Gold Standard**: `20/20 (100.00)`
- **R01..R18 Invariant Suite**: `54/54 PASS (27 Flash + 27 Opus, 0 failures)`
- **54-Track Structural Scan**: `60814 events, 0 blocks > 2 lines, 0 duplicate {\an8} tags`
- **Jevify 20-Scene Dialect/Accent Benchmark**: `2/20 -> 20/20 both_properly_reflected` (`Mean P(both): 0.180 -> 0.597`, `Flash: 0.470 -> 0.661`, `Opus: 0.524 -> 0.593`)
