# GA Generation 0 Baseline (`ga-gen0-baseline`)

- **Official Gold Standard Fitness (10 Movies)**: **100.00 / 100.00** (10/10 Pass)
- **Gen-0 Candidate Mean Fitness**: **83.21**
- **Invariants Passed (`R01..R14`)**: **25 / 27** (6 failures)

## Failing Session Invariants in Gen-0 Baseline
- `R07_moniker_chastity` on `deadpool-2016` (c1369..c1371): got `자, 지금 무대에 / 오를 아가씨는 / 채스티티에게 박수를! / 일명 '아이러니'지` (expected: `must contain '순결'`)
- `R07_moniker_blind_al` on `deadpool-and-wolverine-2024` (c1032..c1037): got `룸메이트인 눈먼 흑인 노인 / 블라인드 알이 그러는데 / 룸메이트인 눈먼 흑인 노인 / 블라인드 알이 그러는데 / 고통을 겪어야 / 나를 안대 / - 기막히네 / - 현명하지? / 이름이 그래? / '블라인드 알'이라고? / 이름이 그래? / '블라인드 알'이라고? / 진짜 앞이 안 보이거든` (expected: `must contain '장님 알'`)
- `R06_conditional_echo_if` on `deadpool-and-wolverine-2024` (c1128): got `세상 고쳐주면 / 나가면 뭐부터 할래?` (expected: `must contain '만약'`)
- `R06_conditional_echo_if` on `deadpool-and-wolverine-2024` (c1134): got `네 세상... / 고쳐주면?` (expected: `must contain '만약'`)
- `R06_conditional_echo_if` on `deadpool-and-wolverine-2024` (c1129): got `세상 고쳐주면 / 나가면 뭐부터 할래?` (expected: `must contain '만약'`)
- `R06_conditional_echo_if` on `deadpool-and-wolverine-2024` (c1135): got `'고쳐주면'이라니?` (expected: `must contain '만약'`)
