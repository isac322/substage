# Korean subtitle style guide

The house standard for Korean video subtitles, modeled on a leading Korean broadcast subtitler and validated on 17 of his videos. It applies to any video and genre. A subtitle renders what the speaker *does* with the line, in the voice a Korean speaker would use in that situation, in as few characters as the meaning allows. It is not a gloss of the English.

Examples: `EN | 직역 (bad) | 좋은 예`; " / " in table examples is display notation for a line break (`\n`), or a cue break where the English has one too. NEVER output literal " / " inside Korean subtitle strings. Use a real newline (`\n`) for two-line subtitles, and only use "/" without spaces for fixed tokens like 24/7, km/h, or 남/여. Names in examples are placeholders. Learn the technique, never the wording.

## Decision procedure (every line)

1. **Who speaks to whom?** Relationship, power balance, the world the video sets (family, office, rigid hierarchy, underworld, royal court), the speaker's attitude right now. This fixes speech level, pronoun and address terms (section 3).
2. **What is the speech act?** Informing, asking, teasing, jabbing, pleading, threatening, joking, reacting. The Korean must perform the same act, even if the words change (section 1).
3. **What must survive?** The plot fact, the punch word, names and glossary forms, a setup a later payoff needs, and any ambiguity the video keeps open.
4. **What can go?** Openers, hedges, tag questions, courtesy formulas, repetition, tag vocatives, vocalizations, and anything the picture or the actor's voice already carries (section 2).
5. **Say it as a Korean would, then cut to the aim.** Read it aloud in the character's voice. If it sounds dubbed, written or textbook, rewrite it, don't trim it.
6. **Fit the screen.** At most 14 characters a line (spaces count), 12 for a one-line cue, two lines at most, broken by sense (section 8).

## 1. Meaning and speech act

### 1.1 Translate the speech act, not the sentence.
Why: the viewer takes the subtitle literally, so a jab rendered as a sincere statement inverts the meaning. In every genre about 1 line in 6 is paraphrased around the intent.
Not when: context leaves the act unclear. Translate plainly, and never add sarcasm or a punchline the scene lacks.
- I thought you'd be more handsome in person. | 실물이 더 나을 줄 알았는데 | 생각보단 못생겼네
- No problem. (answering thanks) | 문제없어요 | 고맙긴요
- Thanks for asking. (nobody asked; pointed) | 물어봐 줘서 고마워 | 참고로 (then state the fact)

### 1.2 Write sarcasm and deadpan as the real attitude: a short counter-punch, sting at the end.
Why: Korean doesn't carry English ironic politeness on the surface. Read literally, it sounds sincere or like a plain command.
Not when: the English is not clearly ironic. Don't make neutral lines snarky.
- Feel free not to. | 제발 하지 마라 | 안 해주면 고맙고
- Oh, great. (things just got worse) | 오, 좋네 | 미치겠네
- Well done. Guess what happens now? | 잘했어요, 이제 어떻게 될까요? | 잘하셨어 / 일을 키우셨네

### 1.3 Keep the content faithful; adapt the form freely.
Why: freedom of form makes the subtitle land; freedom of content is mistranslation.
- Never invent content, motives or jokes. Ambiguous English: pick the reading the scene supports (who speaks, to whom, what just happened).
- An English name used as a nickname, joke or insult: translate its function (the mockery), not just its sound.
- Make implicit situational, nonverbal, or partitive information explicit when a literal Korean gloss would lose the point:
  * Partitive / body metonymy (singular verb with quantifier over a person, e.g., `all/part of you + was`): Korean has no singular/plural verb agreement, so you MUST explicitate the spatial/partitive noun (`데 / 곳 / 쪽`: `Not all of you was asleep.` -> `깨어 있는 데도 있더군` / `딴 데는 깨어 있던데`, NEVER `다 잔 건 아니던데`).
  * Visual posture & action deixis (`Put her down over here` while tying to a chair -> `이쪽에 묶어놔`; `Stay down there` from upstairs -> `올라오지 마`; `You look ridiculous` about a fighting crouch -> `이 꼴로 싸우게?`).
  * Colloquial idioms & rhetorical implicatures (`Oh, I smell what you're steppin' in` -> `뭔지 감이 왔어`, NEVER literal `똥 밟은...`; `Where else do you put one out?` -> `내 몸 말고 딴 데도 꺼?` / `난 온몸이 재떨이야`).
Not when: explicitation would add plot spoilers the English withholds, or metaphorize a plain literal question (`How long was I asleep?` -> `나 얼마나 잤어?`).
- Thank Ganesh while you're at it. | 가네샤에게도 감사해 | 여긴 인도니까 / 가네샤 신께 감사해
### 1.4 Use Korean's question/statement flexibility both ways.
Korean voices taunts, reproaches and disbelief as questions (누가 ~래?, -냐?) and English rhetorical questions as flat statements. In every genre about 1 English question in 10 becomes a Korean statement, and 1 statement in 40 becomes a question.
Why: keeping the English mood sounds stilted.
Not when: the speaker sincerely wants information.
- You two, no one said stop shoveling. | 둘 다, 삽질 멈추지 마 | 너희들 / 누가 삽질 쉬래?
- Hey, you can't talk to my wife that way. | 내 아내한테 그렇게 말하면 안 되지 | 말 막 할 거야?
- Why don't you start seeing with your heart? | 마음으로 보는 게 어때? | 마음으로 봐야지

### 1.5 Answer pushback with a Korean retort form.
Korean pushes back by echoing the other speaker's word: ~긴 무슨 / ~긴 뭐가 ~해, a scolding ~기는, 어련하시겠어 for a sarcastic "Of course you do / Sure", ~는 개뿔 for "X, my ass", 고맙긴 for a deflected thanks.
Why: a plain negation or 당연하지 reads as agreement or a flat correction.
Not when: the denial is sincere and factual. One retort form per exchange at most.
- It's not little. | 작지 않아 | 작긴 뭐가 작아
- Of course you do. (sarcastic) | 당연히 그렇겠지 | 어련하시겠어
- The plan is we're in a cow pasture, that's the plan. | 계획은 소 목장에 있는 거야 | 소 목장에 와서 / 계획은 개뿔이
- Oh, God! You're dull. | 세상에! 너 따분하다 | 재미없기는

## 2. Condensation

The target style is about 10% shorter than a faithful rendering, at every speech rate. It isn't a response to time pressure, so condense even when the cue is long. The longer the English, the harder the cut: a 1–3-word line is rendered in full, a 13+-word line loses about a third. The gap comes from **dropping whole elements**, not squeezing words. Cut the element, then write what is left naturally.

### 2.0 Budget the cue before writing it.
Rough budget, spaces counted: 1–3 English words → about 5 characters; 4–7 → about 11; 8–12 → about 17 (two lines); 13+ → about 20 (two lines). Aim for about 8 characters a line. When a cue won't fit the line limits (8.0), cut an element (2.1–2.7); don't cram.
Why: a cue that fills both lines to the limit reads as machine output.

### 2.1 Drop sentence-initial fillers and reaction openers.
Well / So / Look / Hey / Now / Um / Oh / Anyway / You know get no Korean word (음, 자, 그래, 야, 어) by default; about 1 opener in 5 survives.
How: if an opener survives, it becomes a connective that carries logic (그럼, 근데, 아무튼, 그러니까), never a sound (음, 어).
Why: the viewer hears the hesitation; written down, the subtitle stutters where the actor didn't.
Not when: the hesitation is the content (stalling before a confession). Okay / All right on its own as an answer or a decision is content: 그래, 좋아. A greeting "Hey." is 안녕 / 왔어.
- Listen, I'll talk to the wife. I'll, uh, see if she remembers anything. | 들어 봐요, 아내랑 얘기해 볼게요. 어, 기억나는 게 있는지 볼게요 | 아내한테 기억나는 게 / 있는지 물어볼게요

### 2.2 State it flat: drop hedges, never add intensifiers.
"I think / I guess / I mean / actually / kind of / maybe / probably" need no ~인 것 같아, 아마, 그냥. Add no 정말, 진짜, 완전, 너무, 제발, 대체 the English lacks.
How: move the softness into the ending (-을 거야, -겠지, -려나, -든지, 뭐, 내가 볼 땐).
Why: Korean gets softness from rhythm and endings; mirrored hedges and added emphasis make the line longer and needier than the speaker.
Not when: the uncertainty is the point (a guess, a bad lie): keep one marker. A letter or a narrator's reflection read aloud keeps the writer's hedge.
- I mean, you saved me. | 내 말은, 네가 날 구해줬잖아 | 네가 날 구했어
- She's probably okay. | 아마 괜찮을 거예요 | 괜찮을 거예요
- Now, maybe a little bit. | 뭐, 아마 조금은요 | 약간 있었으려나

### 2.3 Compress repetitions to one or two beats.
Why: three identical beats read as clutter.
Not when: panic or pleading needs exactly two; one flattens the emotion. A chant, a crowd or team call, or a mocking echo keeps its rhythm, so three beats are fine there. A tactical "Go, go, go!" becomes the function word (진입, 진입 / 빨리빨리).
- No, no, no, no. Daddy's fine. | 아니, 아니, 아니, 아니. 아빠는 괜찮아 | 아니야, 아빠 괜찮아
- Please! Please! Please! (thieves mocking) | 제발! | 제발! 제발! 제발!

### 2.4 Merge stacked clauses; keep the point, cut the setup.
Keep the result or the reason, not both; drop echo lines. When the English spells out an action or description the viewer just saw, point at it (이런 짓, 그거, 이런다고) instead of restating it. In a list of three or more, keep every item that is a fact; trim near-synonym or rhetorical triads to two.
Why: English stacks short sentences; Korean folds them into one clause with a connective ending.
Not when: each clause carries a separate plot fact.
- Following someone home and writing down their license plate is tantamount to stalking. | 남의 집까지 따라가서 차 번호를 적는 건 스토킹이나 마찬가지예요 | 이런 짓은 / 스토킹이나 다름없지
- You know, those things are really bad for you. They just build up excess mucus. | 그거 정말 몸에 안 좋아. 점액만 쌓인다고 | 그런 것 좀 먹지 마 / 건강에 안 좋다니까
- There'll be more Christmases, more feasts, more glad tidings. | 크리스마스도 연회도 기쁜 소식도 또 있을 거야 | 크리스마스든 연회든 / 앞으로도 많잖니

### 2.5 Leave pure vocalizations unsubtitled (ko "").
Grunts, Oh / Uh / Ow / Whoa / Shh / Aah, nameless crowd chatter, vocal fills in songs.
Why: the viewer hears them; subtitles take reading time from real lines.
Not when: the sound carries meaning: a refusal ("No!"), a punchline, a reaction the plot depends on, surprise or pain with no other words in the line.
- Ow. Ow. Ow. Ow. | 아야, 아야, 아야, 아야 | (no subtitle)

### 2.6 Replace an explanatory clause with a compact idiomatic phrase.
Machine, HUD and alarm readouts may end in telegraphic nouns. Archaic or ornate English goes into plain modern words; the period feel lives in the endings (3.8), not the vocabulary.
Why: a punchy phrase does in 5 characters what a mirrored clause does in 15.
Not when: it would lose the referent; nominalize only when the subject is on screen.
- It's really nice to have somebody to talk to. | 속 터놓고 얘기할 상대가 있으니 좋네 | 위로가 된다
- Suit compromised. (system voice) | 슈트가 손상됐습니다 | 슈트 손상됨
- I cannot tarry. | 지체할 수 없소 | 꾸물댈 시간 없어

### 2.7 Politeness formulas become the speech level.
"I'm afraid / if you would be so kind / if I may / forgive me / I'd like to / I must say / I beg your pardon" get no 죄송하지만 / 유감이지만 / 실례지만 / 외람되지만. The courtesy goes into the ending (-시겠습니까, -네만, -습니다).
Why: Korean marks deference in the verb; a translated formula adds a clause and sounds like a butler in a dub. About 4 formulas in 5 leave no courtesy word.
Not when: a service script (staff refusing a guest), or the courtesy is the act itself (apologizing; protesting to a superior: With all due respect → 외람된 말씀이지만).
- Would you be so kind as to come with me? | 저와 함께 가 주시겠습니까? | 같이 가시겠습니까?
- Forgive me, but what do you know? | 실례지만 뭘 알고 계신데요? | 그럼 아는 건 뭡니까?
- I'm afraid this facility doesn't allow animals. (staff to guest) | 유감스럽게도 이 시설은 동물을 허용하지 않습니다 | 죄송하지만 동물은 / 반입이 안 됩니다

## 3. Register and address

Each speaker→addressee pair gets one level for the video, decided by the ladder below: the first step that applies wins. Decide from who the two are to each other (kinship, who raised whom, generation, rank, shared history, enmity), never from English politeness words ("sir", "please"; the deference moves into the ending) and never from temperament. Rude, gruff, sarcastic, cold, gentle or polite characters keep the ladder's level; their manner goes into vocabulary (존나, 더럽게, 처먹어요) and exclamations, not into the ending. Age means apparent generation, not literal years; gender alone decides nothing. hae = 반말, haeyo = 해요체, hasipsio = 하십시오체.

1. **Tactical, assistant, or system AI → user: 해요체** (하십시오체 in military/command-net roles). The human user speaking to an AI uses 반말 (step 7 or 10). Only an AI explicitly characterized as a close peer, buddy, or older-sister companion (`longtime_crew_or_buddies`) speaks 반말 under step 5.
2. **Enemies: 반말.** Adults who are enemies by stance (not a passing quarrel), a superior who shows open contempt, a masked fighter to the criminals he fights. An adult enemy is 당신. 존댓말 between enemies is a mock-formal act (3.6).
3. **Own parents who raised the speaker: a family trait.** (A biological parent first met late in life falls under step 4.) Young children, and teens or young adults in warm, casual, modern, or single-parent homes, speak 반말; only children in strict patriarchal, traditional, or period/noble families speak 해요체 (어머니/아버지 in formal homes).
4. **Other elder kin and surrogate parents: 해요체**, even in a warm home: aunts and uncles, grandparents, in-laws, and anyone of an older generation who raised or protects the speaker as a parent would (guardian, foster or found parent, an adult who takes in a child). Address them by kin or role term (아버지, 이모, 삼촌, 아저씨), never 너.
5. **Crews and old buddies: 반말 regardless of rank**, the boss included: a long-standing crew, gang or team, and friends from way back. A junior speaks up to a boss only in a formal institution: 해요체 in an office, school or agency; 하십시오체 in the military, a police unit, a court or a guild.
6. **Younger to senior: 존댓말.** A speaker a generation younger than the addressee, or addressing a mentor, teacher, boss, commander or civil elder, speaks 해요체 (하십시오체 in a rigid hierarchy or a formal role). This holds through intimacy, caretaking, years under one roof, sarcasm, profanity and anger.
7. **Senior to junior: 반말.** Parents, mentors, bosses, commanders, elders to people they raised or know well, and any adult to a child or teen. A man of an older generation uses 자네 (light -게/-나 where natural), not 너, to an adult man he mentors, employs or raised but did not father, and to an old male friend his own age; never to women, children, teens or his own son. Children and teens get 너 or their name.
8. **Comic mockery: 반말 to non-elders.** A character whose running joke is mocking everyone speaks 반말 to strangers, officials and enemies, also when asking favors or pleading. Gruff, rude or sarcastic characters are not this: they speak 해요체 to civil strangers.
9. **Met as adults: mutual 해요체**, between civil strangers, officials, or workplace professionals. Same-generation young adults or romantic partners who bond in private switch to mutual 반말 at their first intimate/private talk as equals; record that as a shift (or use step 10 when they are already an established couple or spouses).
10. **Equals and intimates: 반말.** Friends, siblings, classmates, teammates, established couples and married spouses of one generation (even if `history` is `met_as_adults`); partners since youth.

**Shifts.** A pair's level changes only at a beat the video shows: a role or mask put on or dropped (job interview, disguise, date, mock-formal act), an identity reveal, the first private talk between young-adult equals, a lasting change in the relationship. A passing quarrel is not a shift, and anger never flips a younger→senior pair. Never mirror the other side's level.

- Can you just eat the damn soup and go to sleep? (grown man → old man who raised him) | 수프나 먹고 자 | 수프나 드시고 / 좀 주무시라고요
- You're curious, huh? (crew engineer → crew boss) | 궁금하신가 봐요? | 궁금한가 보네?
- Shall I activate the targeting? (tactical AI → user; companion/buddy AI uses 반말 under step 5) | 조준 기능 켤까? | 조준 기능을 켤까요?
- You need to rest, mister. (girl → unrelated man protecting her) | 너 쉬어야 돼 | 아저씨도 좀 쉬어요
- Yeah, Pop, I'm a lousy son. (adult son → elderly father figure, mid-fight) | 그래, 아빠, 나 형편없는 아들이야 | 그래요, 아버지 / 개떡 같은 아들이죠
- Keep coming, Dad. You're doing great. (child, warm home) | 계속 오세요, 아빠 | 계속 와 / 잘하고 있어
- You really think I did it? (young nurse → detective she met this week) | 내가 했다고 생각해? | 제가 했다고 생각하세요?

### 3.4 Public speech, raids, radio and voice-over each have a fixed register.
News, announcers, PA, officials introducing themselves (FBI의 [이름]입니다) and ceremonies take 하십시오체; a public talk to children (classroom video, PSA) is 반말. A person speaking for themselves to an audience (eulogy, charity appeal, toast) takes 해요체. Police at gunpoint take terse 해라체; radio and command-net traffic follow 4.8. Service talk is 해요체. Fourth-wall asides and voice-over that talks to the audience keep the narrator's in-scene register (an irreverent narrator speaks 반말); a reflective literary memoir uses written 해라체 (대학에 들어갔다).
Why: each setting has a fixed register; drifting down and over-formalizing both sound wrong.
- Police! Put your hands up! | 경찰입니다! 손 드세요! | 경찰이다! 손들어!
- My husband was an ordinary person. (eulogy) | 평범한 사람이었습니다 | 평범한 사람이었어요

### 3.5 Choose pronouns by relationship.
당신: adult strangers, rivals, adult confrontation, a respected colleague outside an upward hierarchy, and spouses, lovers and exes, at any speech level. 너: kids, teens, friends, or deliberate disrespect. 자네: a mature man to a younger man or to a male peer his own age (old friends, veteran colleagues); never to women, never from young speakers. 그대: royal address or song lyrics only. Drop the pronoun among friends and with children. Never 그/그녀 as a default "he/she": use the name, the role, or nothing. Civil upward kinship (own parent, elder kin, surrogate parent) never takes 너/너희/당신, even with warm-home 반말; use omission or a kin/role term. Civil mentor/protege/subordinate exchanges at 해요/하십시오 use omission or the established name/title, not 너/당신. Anger alone does not license 당신 to a parent or mentor. The register table's `pronoun="none"` explicitly forbids surface second-person pronouns; it is not permission to guess one.
Why: an unmarked 너 to a stranger reads as a challenge; 그녀 reads as written prose.
**Third-person referent & deictic resolution (Centering & NIKL `이/그/저` system).** In Korean subtitles (empirically 79% zero-anaphora, 15% explicit name/role/kin NP, 4% adnominal `그 + 명사`): (1) When topic continuity holds and the referent/possessor is unambiguous from the preceding cue or screen, omit the pronoun (`옷은 만년 과장이지만`) or use adnominal `그 + 명사` for a just-stated proposition (`그 조건은 제가 권한 게 아니에요`, `말씀은 이렇게 드리지만`). (2) When the subject or possessor switches, is contrastively focused (`We are on his timetable, not yours` -> `본더먼 일정에 맞춰야 한다고요`; `His terms are not the ones I recommended` -> `트래비스가 내건 조건도 / 제가 권한 게 아니에요`), or competes across a scene/speaker change, restore the character's Korean name, title, or relationship term from CONTEXT (`빌 말대로`, `에디 말이 맞아요`, `블랑 씨는 정중하게 부탁했지만`). (3) Never translate singular English `he/his/him/she/her` as `그쪽` (which defaults to 2nd-person "you" in spoken Korean and flips the referent to the listener), never refer to a co-present colleague/partner on the speaker's own side as `저쪽` (distal/opposing party) or `저 사람`/`저자` (stranger/archaic cur), and use `이분/이쪽` when introducing a co-present colleague (`이분이 새 팀장 케빈이에요`).
- Who are you? And where am I? (to a stranger) | 넌 누구고 여긴 어디야? | 당신 누구야? / 여긴 어디지?
- I can't imagine my life without you! (wife to husband) | 너 없인 못 살아! | 난 당신 없인 / 한순간도 못 살아!
- You came up to town with your girlfriend and met us all. (old classmates, 60s) | 네가 여자 친구를 데려와서... | 자네가 데려와서 / 다 같이 만났잖아

**Highest-Status Addressee Dominance (존자 우선의 법칙).** A mixed group's plural "you" follows the highest-status addressee's address standard. If a parent, elder or mentor is included, never reduce the group to 너희/너네/니들; omit the pronoun or use 다들, 두 분 or the established kin/role term, retaining the table's speech level. A civil public/company audience takes 여러분 with 해요/하십시오, 다들/모두 or omission with 반말, never 너희 or singular 당신. A quoted line, a peer's collective possessive (너희 엄마/집), or a genuinely hostile taunt is not civil upward/group address.
- I invited all you guys. (group includes speaker's mother) | 너희 다 초대했잖아 | 다 초대했잖아
- You all think so? (company audience, 반말) | 너희도 그렇게 생각해? | 다들 그렇게 생각해?
- We care about you, Bill. (civil mentor, 해요) | 우린 당신을 아껴요, 빌 | 우린 빌을 아껴요

### 3.6 Use level play as a weapon.
Mock-honorifics inside 반말 make a taunt (좋아하시네, 어련하시겠어, 대단하시군, 그러셔?); an abrupt shift marks a turn in the scene.
Not when: there is no taunt or turning point.
- Someone's not counting. Six. | 숫자 안 센 놈 있네 | 숫자도 못 세시나? / 여섯
- You have him to thank for this. | 걔 덕인 줄 알아 | 그 친구한테 / 고마워하시지

### 3.7 Drop tag vocatives; keep a name that opens the line to get attention. Never invent address terms.
A name or title tacked onto the end of a line ("…, John." / "…, sir.") is usually dropped. One that opens the line to call someone, or a shouted name, usually stays. Generic man / dude / kid / buddy / guys almost never survives. Add no 아저씨, 이봐, 꼬마, 친구, 얘들아, 선생님 the English lacks.
Why: Korean implies the addressee; extra address terms are among the clearest signs of a dub.
Not when: a shout across distance, a call for attention, picking one person out of a group, a relationship beat or a taunt, or the pair's kin or role term standing in for an English "you", name or "Pop" (아저씬 쉬어야 돼요).
- You've been a serious pain in the ass for me, [Name]. | 넌 정말 골칫거리였어, [이름] | 어지간히 / 골치 아프게 하는군
- Well played, sir. | 잘하셨습니다, 선생님 | 잘하셨습니다
- [Name], you hear me? | [이름], 내 말 들려? | [이름], 들려?
- Trust me, kid. If he wanted to lay you out, he would've. | 내 말 믿어, 꼬맹아... | 진심이었으면 / 상대도 안 됐어

Archaic endings (하오체 -소/-오, 하게체 -네/-게/-일세/-는가) belong only to characters the video sets in an old world: period or royal figures (3.8), grandees of a stylized underworld or establishment (crime lords, guild elders, hotel managers, spy chiefs, dictators), and parody. Assign them per character and keep them the whole video. Never give them to ordinary modern speakers, and a period setting alone does not license them for everyone.
- I have some good news. (mentor to teen, modern) | 자네에게 좋은 소식이 있네 | 좋은 소식이다
- I have no quarrel with him. (old-world crime boss) | 그 사람한테 악감정 없어요 | 난 그자에게 / 악감정 없소

### 3.8 Period, royal and ceremonial worlds: draw a register map before translating.
Sovereign→subjects: archaic 해라체 (-느냐, -거라, -노라, -구나) with 그대(들). Subject or knight→sovereign: 하십시오체 + 폐하/전하. Adult strangers and commoners to each other: 하오체 (이보시오, 잘못 보셨소). A lord or host to a young guest: 하게체 (식사부터 하게). Across a class gap even lovers are asymmetric: the low-born partner speaks 해요체, the noble speaks 반말. A young protagonist speaks ordinary modern 해요/반말. Titles: Sir X → X 경, Your Majesty/Highness → 폐하/전하, my lady → no vocative, honorific vocabulary instead (존함이 어떻게 되시죠?). English archaisms (thee, aye) are never imitated in the words.
- What do you see? (king to nephew) | 뭐가 보여요? | 뭐가 보이느냐?
- Which direction are you headed? (stranger on the road) | 어디로 가? | 어딜 향해 / 가는 것이오?
- I have none to tell, King. (knight to king) | 할 얘기 없어요, 왕이시여 | 폐하께 올릴 / 이야기가 없습니다

### 3.9 Dialect, thick accent & culturally bound accent jokes.
**Unmarked regional accent:** do not mechanically map every British, Southern or Hispanic speaker into a Korean provincial dialect; carry an everyday sociolect through vocabulary and rhythm at the ladder's speech level.
**Plot-critical / comic dialect (`comic_dialect`):** when a speaker's thick dialect or unintelligible pronunciation is itself the butt of a joke or a communication barrier in the scene (Gambit's Cajun English in *Deadpool & Wolverine*: Deadpool mocks it via *Sling Blade*, asks if his dialect coach is the Minions, translates what he is trying to say), or the English subtitle uses eye-dialect (`we done be knowing`, `wit'it`, `I ain't know`, `biggum hands`, `Ain't not a wonna`), render those lines with unmistakable earthy Korean dialect markers (근디, ~거시여, ~제, ~뿌서, ~믄, 몬, 거시기, 울 엄니, ~슈) while keeping the speaker's consistent speech level (반말 for Gambit). Never let review smooth these endings back to standard Korean.
**Culturally bound accent metaphors:** when an Anglo-American pop-culture title or name is used purely as shorthand for someone's accent (*Sling Blade* = thick Southern drawl; *(IN BRITISH ACCENT) Basil Fawlty* = stereotypical British accent; *Foghorn Leghorn* = Southern drawl), localize the reference so a Korean viewer gets the accent joke at once.
**Manner-of-speaking SDH and regional slang:** respect `[In New York accent]` / `(IN BRITISH ACCENT)` delivery tags (the observation note carries them as `accent_sdh=`): Peter faking an adult New York voice to reassure passengers → 안녕하세요 / 이제 안심하세요; Delmar's bodega deference → 대령합죠. Translate British/Yiddish slang by meaning (you mug → 이 호구야 / 등신, oy vavoy → 세상에나), never phonetically (머그시, 오이 바보이).
Why: when the accent is the joke, standard Korean deletes the joke; when it isn't, a provincial dialect invents one.
- Have you seen Sling Blade? | 「슬링 블레이드」 봤어? | 남부 억양 지리네 / 다시 말해줄래?
- We done be knowing that lid ain't comin' off. | 그 뚜껑은 안 열릴 거야 | 근디 그 뚜껑은 / 안 벗겨질 거시여
- Biggum hands close, ain't not a wonna get in. | 큰 손이 닫히면 아무도 들어가요 | 큰 손이 딱 닫히믄 / 아무도 몬 들어가지
- I hate to hurt you, but is your dialect coach the Minions? | 상처 주긴 싫은데 미니언즈가 코치야? | 상처 주긴 싫은데 / 사투리 선생이 미니언즈야?
- (IN BRITISH ACCENT) Is it Basil Fawlty? | 배질 폴티야? | 영국인답게 / 배질 폴티야?
- Get back here, you mug! | 이리 와, 머그시! | 이리 와, 이 호구야!

### 3.10 Mr./Ms. and endearments.
Mr./Ms. + surname → surname + 씨, or dropped as a vocative (3.7). With a rank, role title + 님 (점장님, 국장님). An official addressing a young woman: surname + 양. 미스터/미스 only as a mock-formal villain tic; never 선생 for "Mr." Endearments (baby, honey, darling, sweetheart) are dropped in most lines; couples who keep one use 자기야, a spouse calling to a spouse 여보. A character's signature endearment may be transliterated and kept every time.
- Mr. [Surname] wants to meet us. | 미스터 [성]가 보재 | [성] 씨가 / 우릴 보고 싶대
- Not this, darling, the house. | 이거 말고, 자기야, 집 | 얼굴 말고 집

## 4. Spoken endings and anti-translationese

### 4.1 Drop English tag questions and confirmation checks.
"…, right? / okay? / remember? / huh?" are rhythm, not 알았지?, 그치?, 맞지?. Most become statements; the rest fold the tag into the ending (-지?, -다며, -죠?).
Why: they make every line sound needy and add a beat to read.
Not when: the speaker really waits for an answer or forces a promise.
- Just give me five minutes, okay? | 5분만 줘, 알았지? | 5분만 줘
- But even you said she was hiding something, right? | 그래도 너도 뭔가 숨긴다고 했잖아, 맞지? | 그래도 뭔가 / 숨기고 계셨다며

### 4.2 -잖아 only for knowledge both sides share.
Shared fact, recall or retort only. New information: say it plainly. Just learned ("turns out / apparently / I heard / I noticed"): -더라(고), or -더군 for an older voice. "You said / remember?": -댔잖아. Terse thrillers barely use -잖아 at all.
Why: -잖아 means "you know this"; on new information it sounds presumptuous, and in bulk it is a tic.
- Einstein said time was relative, right? | 아인슈타인이 시간은 상대적이라고 했지, 맞지? | 아인슈타인이 / 시간은 상대적이랬잖아?
- Turns out our founder's tailor was one of us. | 알고 보니 설립자의 재단사가 우리였어 | 설립자의 재단사가 / 우리 쪽이었더라고

### 4.3 -거든 sparingly: a reason the listener lacks, or a retort.
Use -거든 for a fact the listener doesn't have, given as an aside or an answer to "why?", or for a comeback. An ordinary "because" is -니까/-어서 or a restructured line. At most one -거든 in an exchange. (Conditional -거든, "if", is a different ending and is fine.)
Why: overused, -거든 makes every character sound smug.
- Our lead singer just quit. | 우리 보컬이 방금 관뒀어 | 우리 보컬이 관뒀거든
- Thanks, but I already have a date. | 고맙지만 난 이미 파트너 있거든 | 고맙지만 파트너 구했어
- Because I sell drugs. | 난 마약을 팔거든 | 난 마약을 파니까

### 4.4 "But" is dropped by default; 하지만 only for a weighty pivot.
Fold the contrast into an ending (-는데, -지만), use 근데/그래도 in chat and banter, or leave it implied. Keep 하지만 for a threat, a moral point, a grave explanation, a monologue or a formal speech (about 1 serious line in 4).
Why: sentence-initial 하지만 in casual talk is essay register.
- But Charlie is on the phone with his girlfriend. | 하지만 찰리는 여자친구랑 통화 중이에요 | 찰리가 애인과 / 통화 중이에요
- But he's bad for business. | 하지만 사업엔 방해돼 | 그래도 / 사업에 안 좋으니
- But if you kill him, he wins. (grave) | 근데 죽이면 걔가 이기는 거야 | 하지만 죽이면 / 저놈의 승리야

### 4.5 Write spoken forms, not written ones.
-는 거, 이거, 그거, 근데 are the default, not -는 것, 이것, 그런데. 것 같 is acceptable, especially from older, formal or deadpan speakers; don't force 거 같, since the real problem is usually the hedge itself (2.2). Storybook endings (-단다, -란다, -렴) belong to a parent or elder speaking warmly to a child, or to mock-paternal irony (an anti-hero talking down to a kid), never to peers or teens among themselves. In-scene exclamatory -다 is natural spoken 반말 (완전 딱이다, 그만 가야겠다).
Why: 것 forms in young or casual speech sound like reading aloud.
- I'm starting to think there's a reason I'm in this cab. | 이 택시를 탄 이유가 있는 것 같아 | 이 택시 탄 이유가 / 있는 거 같아

### 4.6 Rebuild the sentence in Korean order, not English syntax.
No ~에 의해 passives, no ~에 대하여, no mirrored subject–object order or chained English clauses. Pre-modify the noun or postpose a clause the Korean way.
Why: English clause order on a Korean line is the core of 번역투.
- There's a big machine called a projector. Inside, there's a big bright light... | 영사기라는 큰 기계가 있어. 안에 아주 밝은 불빛이 있어서 | 큰 조명이 들어있는 / 영사기란 기계가 있는데

### 4.7 Cool, older or hard-boiled observers take -군 / -더군 / -나 보군.
"Time to X / I guess / Nice X / Looks like" from a cool, older, villainous or deadpan speaker.
Why: -네요 or 것 같아 makes these figures sound chatty or unsure.
Not when: teens and warm chat use -네.
- Time to make the tacos. | 타코 만들 시간이야 | 타코 만들 시간이군
- I guess dubstep never dies. | 덥스텝은 안 죽는 것 같아 | 덥스텝은 / 죽지도 않나 보군

### 4.8 Radio, tactical and dispatch talk: plain -다, -라 and nouns, no "!".
"Copy / Roger" → 알았다 (subordinates: 알겠습니다). Status reports end in plain -다 or a noun; unit orders take -라 or 바란다; "Over" → 이상.
Why: chat endings (-어/-요) on the radio sound like civilians playing soldier.
Not when: the same people off the radio.
- Target is now eastbound. | 타깃이 지금 동쪽으로 가고 있어요 | 동쪽으로 이동한다
- We're six minutes out! | 6분 후에 도착해요! | 6분 전
- All units, please respond. | 모든 대원들 응답해 주세요 | 전대원 출동하라
- Clear! (room sweep) | 깨끗해! | 이상 무

## 5. Profanity

### 5.1 Translate the heat of the line, not each swear token; save strong words for peaks.
Expect a strong Korean swear on about 1 in 4 *fuck* lines (about 1 in 2 in an all-out profane comedy, about 1 in 8 in a grim crime video). Dropping the token does not mean dropping the heat: move it into crude-register verbs and nouns (뒈지다, 처-, 아가리/주둥이, 대가리, 꼴통, 등신, 개소리, 튀어, -고 지랄이야), an intensifier phrase (환장하겠네, 미치겠네, 더럽게), or a bare forceful imperative. Fully neutralize a hot line only when the swear is pure filler.
Why: a Korean swear on screen hits harder than an English one heard in passing; at English rate it reads as nonstop screaming and the peaks lose their punch.
Not when: the line is the punchline or the climax. Use the strongest word the video and speaker allow; an infix gag ("Name fucking Name") keeps 씨발 in the middle.
- Shut the fuck up. | 닥쳐, 씨발 | 아가리 다물어
- They're gonna fuckin' kill me! | 날 씨발 죽일 거야! | 이러다 뒈지겠어!
- Sam fucking Lee. (the comic beat) | 샘 리 새끼잖아 | 샘 씨발 리

### 5.2 Set the swear ceiling by the video's world and the speaker, not the English word or the rating.
Family, teen and gentle drama: around 젠장, 망할, 빌어먹을, 염병. PG-13 thrillers and action: 젠장, 망할, 빌어먹을, 망했다. An R rating is not a license: crime, gangster and assassin videos usually sit at 젠장/썅/새끼/개자식 plus crude verbs. 씨발 and 좆 cluster in outright profane comedy, and even there mostly in the mouth of the one foul-mouthed character. Formal, ritualized worlds (an assassins' guild, a period court) stay near clean even in R videos. The VIDEO rating, tone and PERSONAS in CONTEXT set the ceiling.
Why: "shit" or "goddamn" in a family video sits at the 젠장 level for a Korean viewer; 씨발 breaks the tone.
- Shut up, Chad, goddamn it! (family drama) | 닥쳐, 채드, 씨발! | 닥쳐, 채드!
- Asshole. (R action, formal underworld) | 개새끼 | 망할 놈

### 5.3 Turn intensifier profanity into the ending and word choice.
In "what the hell / who the fuck…" the swear is emphasis: insert no 대체 or 씨발. Three in four become plain questions.
Why: a mechanical 대체/씨발 is a calque, stiffer or harsher than a Korean speaker.
Not when: the curse is the content (aimed at someone, real fury). Keep a strong equivalent (닥쳐, 염병할).
- Who the hell are you? | 대체 누구신데요? | 댁들은 누구신데?
- What the fuck is that? | 대체 씨발 저게 뭐야? | 저게 뭐야?
- Where the fuck were you? | 대체 씨발 어디 있었어? | 어딜 처나갔었어?

### 5.4 Collapse repeated swears to one, or into the emotion behind them.
Not when: a deliberate comic beat built on the repetition.
- Shit! Shit! Shit! (panic) | 젠장, 젠장, 젠장! | 어떡해! 어떡해!
- Fuck! Fuck! (crime thriller) | 씨발! 씨발! | 썅!

### 5.5 Vary the vocabulary; coin creative insults for creative insults.
Why: when an English insult is inventive, the invention is the joke; a stock swear or literal gloss kills it.
Not when: the original is a plain swear.
- What the shit-biscuit! | 이런 개떡 같은! | 이런 씨버러먹을!
- And by dance, I mean… | 춤춘다는 건... | 춤은 개뿔

### 5.6 Localize sexual jokes into Korean crude idiom; body-part insults become plain insults.
Sex acts: 떡치다, 딸치다, 자다, not clinical or transliterated words (성관계, 자위, 블로우잡). A body-part word used as an insult (dick, prick, dickhead, cock, pussy) becomes a plain Korean insult (재수 없는 놈, 머저리, 찌질한 새끼, 망할 놈, 등신). When the body part is literally the topic, use 거시기/물건/내 거, or 좆 in a profane comedy. Keep the joke's mechanics.
Not when: a milder word carries the joke; never escalate to anatomical slurs.
- You fucking dickhead! | 이 좆대가리 새끼! | 이런 머저리!
- more than you love vagina | 보지보다 | 섹스보다

### 5.7 놈/새끼 only when the speaker means contempt.
Neutral "guy / these guys": 사람, 친구, 걔, 그 사람들.
Why: a hero who calls strangers 놈 becomes rude.
- It's gotta be the guy I saw… | 그놈이야 | 그 사람 같지?
- Fucking mutant. (real contempt) | 좆같은 돌연변이 새끼 | 염병할 돌연변이 놈

### 5.8 Teen and slang voice comes from endings and 완전, not slang words; don't add profanity to clean lines.
대박, 쩔다, 겁나 are nearly absent even in teen videos. Whoa / cool / dude / bro are usually dropped or rendered plainly (알았어, 잘됐네, 잘 지내냐?). For admiration in adult videos use 죽이다/죽이는 or 끝내주다. Profanity on a clean English line is almost never added; the exceptions complete a cut-off swear or an explicit f-word pun.
- Whoo! That was crazy. (teen) | 대박 미쳤다 | 완전 끝내줬어
- Dad, you're old, man. (teen) | 아빠 늙었어, 진짜 | 완전 아저씨라니까
- We need to build a fucking team. | 씨발 팀을 짜야 돼 | 죽이는 팀을 꾸리면 돼

### 5.9 A lone expletive is 젠장 or 썅, not 씨발; a swear of praise becomes praise.
Why: an outburst alone on screen is a reflex, not an attack; 씨발 overstates it.
Not when: the video's foul-mouthed character at a comic peak. A swear said in delight or admiration ("Damn.", "Holy shit!") is rendered as admiration (우와, 죽이네).
- Fuck! | 씨발! | 젠장! / 썅!
- Oh, shit. (it's over) | 아, 씨발 | 망했다
- Shit. (impressed) | 젠장 | 죽이네

### 5.10 Give each aimed formula its Korean formula, and keep a character's signature swear fixed.
"Fuck you" → 조까; "fuck off / get the fuck out" → 꺼져, or a crude movement verb (튀어, 처엎드려); "go fuck yourself" → 나가 뒈져. When one character's catchphrase is a swear, pick one Korean signature and reuse it in every line, including lines that talk *about* the word.
Why: a catchphrase is characterization; a different Korean swear each time erases it.
- Fuck you, [Name]! | 엿 먹어, [이름]! | 조까, [이름]!
- Get the fuck down! | 씨발 엎드려! | 처엎드려!
- Motherfucker! (one character's signature) | 개새끼! / 씨발놈! / 새끼야! (varied) | one chosen form, every time

### 5.11 British swears are mild; render the attitude.
bloody / bollocks / wanker / twat / piss sit at 짜증 / 재수 없다 / 머저리 level, not at 씨발/좆 level. "Bloody" is usually nothing.
- I'm tired of the bloody anthems. | 빌어먹을 국가풍 노래 지겨워 | 진지한 곡에 지쳤어
- Wanker. | 딸딸이 새끼 | 재수 없어

## 6. Humor, references and wordplay

### 6.1 Keep real-world names; turn obscure everyday ones into their Korean category.
Keep celebrities, videos, songs, shows and famous brands (most real references survive, even obscure ones that carry the joke). Videos and shows use the Korean release title in quotes. When a local chain, app, venue, institution or product a Korean viewer won't know is only a stand-in, write the plain category (택배기사, 명문대, 대기업, 공항, 술집, 음료수). If the name is used as a verb or a descriptor ("Scooby-Doo this"), write what it means. If recognizing the name is the joke, add a one-word category tag (노래, 영상, 명대사, 흥행), never an explanatory clause. Swap in a different Korean image only for a throwaway gag, only when an equivalent of the same register comes to mind at once; that is rare.
Why: an unknown brand reads as noise. A celebrity or a title is usually the joke itself.
Not when: the plot needs the name later, or the name is the butt of the joke. Never relocate the setting (no 김밥 in Queens).
- You were working at Heathrow before we gave you a chance. | 히드로 공항에서 일하고 있었잖아 | 우리가 기회 주기 전엔 / 공항에서 짐 날랐잖아
- …you impersonated the Postmates driver on the phone? | 포스트메이츠 배달원 흉내 | 너 전화로 택배기사 / 흉내 낸 적 있잖아?
- Do you want to build a snowman? From Frozen… | '프로즌'의 '눈사람 만들래?' | '겨울왕국'의 / '나랑 눈사람 만들래?'

### 6.2 Rebuild sound jokes in Korean sound when one comes; otherwise translate what the pun does.
Rhymes, puns, chants, acronyms, pun-names and mock spellings: make a Korean version with the same mechanism (a Korean acronym, a Korean word blend), keeping the rhythm and the payoff word. No sound game comes: drop the pun and write the jab or meaning it delivers; never a literal pun that doesn't work, never a translator's note. Spelling jokes about a name ("with a K") keep the Latin letter; a mispronounced name the video mocks keeps the mispronunciation.
Why: a literal version keeps the content and loses the reason anyone laughed.
- I'm the Shocker. I shock people. | 난 쇼커다, 충격을 주지 | 내가 쇼커다! / 쇼킹하지?
- Never let anyone describe you as "H.K.L.P." / "Holds knife like pen." | 'H.K.L.P.'라고 불리면 안 돼 | '나펜사'란 소리는 / 듣지 마
- That extra "A" stands for "ass," yours, between mine and a bullet. | 추가된 A는 '엉덩이'의 A야 | 대신 총 맞으라고 / 비싼 돈 내는 거야

### 6.3 Translate descriptive labels and mocking nicknames into everyday Korean; transliterate established aliases.
A label that *describes* (a mode, a ritual, a rank, an institution, a nickname built on a trait) gets the shortest everyday Korean word, often a -이 nickname for people. When a later cue questions or explains a nickname by its literal meaning (`"Blind Al" -> "Well, she's blind"`, `"Chastity" -> "Irony"`), you MUST gloss the trait in Korean (`장님 알`, `순결이`) so the explanation makes sense; phonetic transliteration (`블라인드 알`, `체스터티`) breaks the dialogue. Hero and villain aliases whose literal meaning is not unpacked in dialogue stay transliterated. Once chosen, keep the label every time.
Why: labels return as running gags; a clunky first rendering flattens every repeat.
- "Training wheels protocol"? Turn it off. | '초보자 프로토콜'? 꺼 | '왕초보 모드'? / 꺼버려
- Say hello to "The Grump." (from grumpy) | '그럼프'한테 인사해 | 심술이한테 인사해야지
- [Name], excommunicado. | [이름], 엑스커뮤니카도 | [이름], 파문

### 6.4 Render idioms by their meaning; keep comic images that work in Korean.
Dead English idioms: plain spoken meaning, by far the most common choice. A Korean saying or set retort (개뿔, 뜸 들이다, 궁지에 몰리다) only when it means the same thing in the same tone. Vivid comic similes and absurd comparisons: keep the image if a Korean viewer gets it instantly. Quoted mottos and maxims: a terse -다 sentence in quotes.
Why: a literal English image is nonsense in Korean, and a forced Korean proverb sounds like a translator showing off.
Not when: the image is transparent in Korean or world-famous (fight dirty, lab rats), the metaphor is coined by the video and pays off later, the video comments on the wording, or it is a song lyric.
- I think you're very much barking up the wrong tree. | 엉뚱한 나무에 대고 짖고 계시네요 | 아직 모르시나 본데
- I understand some people get cold feet on their wedding day. | 결혼식 날 발이 차가워지는 사람도 있다더니 | 결혼식 날에 쫄아서 / 튀는 놈들 있다더니
- You're about as useful as a condom in a convent. | 수녀원의 콘돔만큼 쓸모없어 | 수녀원의 콘돔도 / 너보단 쓸모 있겠다
- How you do anything is how you do everything. | 하나를 하는 방식이 모든 걸 하는 방식이다 | '한 가지를 대하는 태도가 / 만 가지를 대하는 태도다'

### 6.5 Comic sung jokes, chants and gag signs: subtitle them, rebuilding the rhyme.
Why: when a character sings a parody, rhyme game or demonstration hook on screen, the laugh is in the sound-play; a blank or a flat medical/dictionary gloss loses it entirely. Rebuild the shared phonetic ending across the chain ('-이카 / -니까', '-카', '-야', '밥바루 루밥'), keep spoken replies ('정답!') in spoken register, and drop pure vocable cues ('Oy, oy, oy', 'Bom, bom, tsk', 'Shoop shooop ba-doop').
Not when: background soundtrack, credit songs, off-screen transition humming tails, or any lyric sitting under foreground dialogue or under on-screen text.
- Kleenex-ica, Windex-ica ♪ / She's sexy-ca, oh, boy ♪ | (no subtitle) | 크리넥사이카, 윈덱사이카 / 그녀는 섹시하니까
- Pneumonia, dyslexia, leukemia ♪ | 폐렴, 난독증, 백혈병 | 알파카, 알라스카, 남아프리카!
- We live in Arizon-ia ♪ … stony-a ♪ | (no subtitle) | 우리집은 애리조니아 / 텅텅 빈 황무지야
### 6.6 Performed songs: subtitle every lyric with a leading ♪, for meaning, in a lyric voice.
When a song is performed on screen as story (concert, audition, rehearsal, recording, a character singing to someone, a song quoted for a joke), subtitle each lyric cue. Translate the meaning, not the rhyme, in a short sung register independent of the singer's speech level: lyrical 반말 by default (즐기며 살아), soft 해요 or -나요 for ballads, pleas and songs to a parent or a beloved (모르시나요?). Lyrics are the one place 그대 and 그녀 are natural. Drop ooh/yeah fills, transliterate nonsense hooks, collapse repeated hooks, and render a repeated chorus identically every time. Song titles stay transliterated; video titles use the Korean release title.
Format: one "♪ " at the start of the cue, never at the end or on line 2, no ellipsis. The pipeline adds the mark to subtitled lyric cues, so write the lyric text itself without ♪.
Why: in a music scene the lyrics are the story; a consistent ♪ tells the viewer it's sung, not said.
Not when: background or credit songs get no subtitle at all; a lyric under dialogue (subtitle the dialogue); a song being composed in fragments (subtitle it as speech).
- Love of my life, can't you see? ♪ | 내 사랑, 모르겠어? | ♪ 내 평생의 사랑 / 모르시나요?
- Mama, ooh, ooh, ooh, ooh ♪ | 엄마, 우, 우, 우 | ♪ 엄마
- If I'm not back again this time tomorrow ♪ | ♪ 내일 이 시간까지 안 돌아와도 ♪ | ♪ 내가 내일까지 / 돌아오지 않더라도

## 7. Exclamations and set phrases

### 7.1 Translate an exclamation as the emotion behind it; ration 세상에/맙소사.
When "Oh my God / Jesus / Christ" opens a sentence, drop it and let the sentence carry the alarm, or use the matching reaction: panic → 어떡해, exasperation → 미치겠네/환장하겠네/돌아버리겠네, disgust or pain → 어우/아퍼, relief ("Thank God") → 다행이다/살았다. "Jesus fucking Christ" → 환장하겠네 / 이런 미친, never 맙소사 plus a swear. A standalone line of pure shock or awe ("My God." / "Jesus.") may take 세상에 or 맙소사.
Why: as a reflex, the dictionary oath sounds dubbed and hides whether the speaker is scared, delighted or fed up.
Not when: the whole line is the stunned beat (shock, awe, a horror reveal). Even then vary it, and keep 세상에/맙소사 to about 1 oath line in 4.
- Oh my god, you're bleeding. | 세상에, 피 나잖아 | 너 피 나잖아
- Jesus Christ. (fed up, a third interruption) | 맙소사 | 미치겠네

### 7.2 Almost never start a line with a reflex 아, or 어,
Drop the "Oh / Ah / Uh / Hey" or use a reaction with meaning.
Why: line-initial 아,/어, is filler the audio already carries.
Not when: 아, 맞다 / 아, 진짜 where the 아 carries sudden recall or exasperation.
- Oh, hey! | 어, 왔냐! | 왔어?
- Oh, no. | 아, 안 돼 | 안 돼

### 7.3 Choose a set phrase by its function in the scene, not its literal sense.
One fixed equivalent produces a stiff, wrong-situation line.
- **Come on** (almost never literal): hurry → 어서/빨리/가자; protest → 이러지 마/됐어/왜 이래; plead → 제발; cheer → 힘내/할 수 있어; otherwise drop.
- **Sorry**: apology → 미안/죄송; condolence → 유감이야/안됐다; "Sorry? / Excuse me?" → 뭐라고?/네?; passing or leaving → 지나갈게요/실례할게요.
- **Welcome (back) to X** → X에 잘 왔어/잘 오셨습니다; 환영합니다 only on signs and in broadcasts; figurative "Welcome to fugitive life" → 도망자 신세군.
- **Of course**: permission → 그러세요/그러시죠; agreement → 당연하지/그럼; sarcasm → 어련하시겠어 (1.5); 물론 only as a connective inside a sentence.
- **Thanks / You're welcome**: most thanks are 고마워; a refusal ("No, thank you") → 괜찮습니다; to a team → 수고했어. "You're welcome / No problem" → 고맙긴, 뭘; 천만에 or 별말씀을요 only from a formal servant or butler type.
- **Trust me / I promise**: the assurance itself (진짜야, 꼭 ~할게, 걱정 마).
- I promise, I've got this all under control. | 약속할게요, 제가 다 통제하고 있어요 | 걱정 마세요 / 금방 해결하니까
- Guys, you do not want to do this. Trust me. | 그러지 마세요, 날 믿어요 | 이러다 후회해요 / 진짜예요

### 7.4 Ceremonial and command phrases: use the fixed Korean formula for the situation.
"My condolences / sorry for your loss" → 고인의 명복을 빌어 / 애도를 표합니다; "Very well" → 좋다/알겠습니다/그러지; "Cheers" (toast) → 건배; "Get down!" → 엎드려!; radio procedure follows 4.8. "Move! / Go, go, go!" have no fixed form: name the action this scene needs (비켜, 뛰어, 밟아, 진입해, 이동해).
Why: in action, spy and period videos these lines carry rank and procedure; a literal rendering (매우 좋다, 복사) is wrong or comic.
Not when: the phrase is used ironically (1.2, 1.5).
- Move! (blocked doorway) / Move! (getaway car) | 움직여! | 비켜! / 밟아!

## 8. Layout, punctuation and on-screen text

### 8.0 Line length and breaks: at most 14 characters a line, 12 for a one-line cue, two lines at most. Break by sense.
Count spaces. Above 12 characters, use two lines; the median line is about 8 characters. Break where a reader would pause: after a vocative, an opener or a finished sentence, or before the verb phrase; never just to balance the lines. When a cue holds two sentences, put one on each line with no punctuation between them; only a very short pair (about 10 characters or fewer) stays on one line, joined with a space, "?" or "!". A vocative or a one-word answer may sit alone on line 1. Never end a line or a cue with a comma or a period, and never put ". " mid-line.
Why: a long single line makes the eye travel, and the line break already does the work of the comma or period. Almost every official line is 14 characters or fewer.
- Uh, I gotta go. I'll be back. | 저 가봐야겠어요, 금방 올게요 | 가봐야겠어요 / 금방 올게요
- Scatter and converge. Get that case! | 흩어져서 포위해, 가방 확보해 | 흩어져서 포위한다 / 가방 확보해
- Sam! What was the first rule? | 샘, 첫 번째 규칙이 뭐였지? | 샘! / 첫 번째 규칙이 뭐지?
Reading budget: about 4 Korean characters per second of screen time (spaces not counted) is typical, about 6 is busy, over 8 is rare. Slow, formal videos sit lower.

### 8.1 Multi-cue sentences, Korean SOV inversion across cues, and ellipsis "...":
Write three periods "..." only for a real cut-off, trail-off or interruption, once per cue, at the end of the words it cuts. Never open a line or a cue with "...", and never copy English "..." that only shows a sentence continuing. Aim for 5% of cues or fewer.
When one English sentence runs across two consecutive cues, decide by its syntactic structure:
1. **Right-branching English modifier / PP in Cue 2 (`REVERSE_SOV_PERMITTED` -> Mandatory Korean SOV Reordering)**:
   - English (SVO, head-initial) puts the main verb and object in Cue 1 and a prepositional phrase (`with / for / from / in / into / between / of / by / through`), relative clause, or instrument in Cue 2. Korean (SOV, head-final) MUST place modifiers and postpositional phrases BEFORE the main verb.
   - NEVER finish the English main verb at the end of Cue 1 and leave Cue 2 as an English-order afterthought fragment (`그것도 ~랑!`, `~에서요`, `~까지요`).
   - Instead, move the main verb from English Cue 1 to the end of Korean Cue 2:
     * **Full SOV Inversion**: Put Cue 2's modifier/PP + noun in Cue 1 (ending with an adnominal/case particle `~의`, `~에서`, `~로`, `~에`, `~하나`), and put Cue 1's main verb predicate in Cue 2.
       - You couldn't even save a relationship… / with a goddamn stripper! | 연애 하나 못 지킨 놈이! / 그것도 망할 스트리퍼랑! | 스트리퍼 따위와의 관계 하나 / 못 지켜내는 새끼가!
       - you are under arrest by the TVA… / for too many crimes to list. | TVA가 당신을 체포한다 / 죄목은 셀 수도 없다 | 시간 변동 관리국에서 수많은 범죄 혐의로 / 당신을 체포한다
       - To distribute gene therapy discreetly through everything... / from sweet drinks to breakfast cereals. | 유전자 치료제를 몰래 섞었어 / 단 음료부터 아침 시리얼까지 | 음료수부터 시리얼까지 모든 식품에 / 유전자 치료를 적용해 몰래 퍼뜨렸고
       - Okay, so how do we calculate linear acceleration... / Between points a and b? | 선가속도는 어떻게 계산하지? / A와 B 지점 사이에서? | 그럼 A와 B 사이의 / 선가속도 계산법은?
     * **Verb Deferral (when Cue 1 has a 2-speaker `-` dash or long subject/object)**: Leave Cue 1's object open (`~을/를/은/는`) and put Cue 2's modifier + Cue 1's main verb in Cue 2.
       - -It looks great. -And you decimated dozens of my men… / with the exhumed corpse of a hero. | - 보기 좋네 - 내 부하들을 몰살했지 / 영웅의 시체를 파내서 | - 보기 좋잖아 - 내 부하 열댓 명을 / 파묘한 영웅의 유골로 죽이고
2. **Dramatic reveals, hesitant pauses, and coordinate clauses (`DRAMATIC_REVEAL_KEEP_ORDER` -> Preserve Order)**:
   - When Cue 1 is a hesitant setup (`Said I was...`, `speak to this...`) before a 1-2 word reveal/name in Cue 2 (`unstable.`, `Spider-Man.`), or when Cue 2 is a coordinate clause (`and / but / so`) or a `that / if` clause after a self-contained statement: preserve the chronological cue order so the punchline/reveal is never spoiled in Cue 1.
   - So you gotta hold onto love… | 그러니 사랑을... | 그러니까 사랑을 / 붙잡으란 말야
   - Said I was... / unstable. (two cues, reveal) | 불안정하다고... / 내가 그랬지 | 내가... / 불안정하다고 했지
   - Oh, gosh, I'm sorry. I thought this was a-- | 미안, 이게 그건 줄 알고 | 맞다, 미안해 / 내가 또...
3. **Multi-cue Discourse Units (`2–5 cues`) and Hesitation + Metalinguistic Completion (`a... -> Let's call it X / Think of it as X`)**:
   - When a speaker's thought spans 3–5 cues or trails off on an article/setup (`we'd like to offer you a...`) before completing the thought with a metalinguistic rephrasing (`Let's call it a favor.`, `Think of it as...`, `Let's say...`), translate the whole discourse unit as ONE coherent Korean speech act first, then slice it across the cues.
   - Never invent a placeholder object noun (`선물을...`) to fill an unfinished `a...` cue, and never calque English framing nouns (`좀 더 실질적인 감사의 표시`, `호의라고 해두죠`). Leave the hesitation cue open (`감사의 표시로 저희가` / `부인께...`) and fold the metalinguistic completion into the natural Korean predicate (`도움을 드리죠` / `도움을 하나 드린다고 생각하시죠`).
   - Resolve pronouns (`that`, `it`, `he`) and omitted subjects/objects across the preceding 2–3 cues so short replies and rhetorical questions (`Am I meant to find that reassuring?` → `그런다고 내 마음이 편해지겠어요?`) keep their exact referent and speech level.
### 8.2 Speaker dashes only when two distinct speakers remain in the same displayed cue.
Use "- " with a space on both lines, one speaker per line. Never put a dash inside one speaker's turn or on a one-line cue. One speaker's long speech or multiple sentences still gets no speaker dashes. If an incidental backchannel is omitted, or the speakers' turns appear in separate cues, leave each single-speaker cue without dashes. An unknown speaker label alone does not establish a speaker change.
Keep an essential answer to a question, or agreement that changes what the viewer understands, as the responding speaker's own turn; its wording may be compressed. A word such as "Yeah" or "Mm-hmm" can be an answer, meaningful agreement or an incidental backchannel: decide by its function in context, never by the word alone. You may omit a backchannel that adds no meaning needed to follow the exchange.
If both retained turns are long, put them in separate cues. If English uses name labels, add dashes only when two distinct speakers remain in the same cue. Aim to keep the pair within about 8 characters of each other: shorten the long turn without losing meaning, never pad or erase the short retained turn to balance the lines.
Why: two voices run together read as one speaker; a short answer can carry the meaning of the exchange.
- Got it? / Yeah. (two speakers; essential answer) | 알겠어? 그래 | - 알았어? / - 네
- It's similar to an organ transplant. / Huh. (incidental acknowledgment, no surprise) | - 장기 이식이랑 비슷해요 / - 흠 | 이건 장기 이식과 / 비슷해요
- You are ready. / I'm not. (name-labelled source) | 넌 준비됐어, 아냐 | - 넌 준비됐어 / - 아냐

### 8.3 Titles, quoted words and text read aloud go in straight single quotes '...', never angle brackets.
Videos, songs, books, shows, words discussed as words, and quoted speech; don't turn quoted speech into indirect paraphrase. When a character reads or recites text (a letter, inscription, poem, report, or a quote longer than one cue), wrap each cue of it in quotes. One quote style for the whole video: straight single quotes (the pipeline normalizes curly and double quotes to it).
Why: players treat "<...>" as a markup tag and delete the title, breaking the line; quotes mark a title or quotation at a glance.
Not when: scare quotes for mere emphasis.
- Have you seen 127 Hours? | <127시간> 봤어? | '127시간' 봤어?
- He said it was a superhero video. (quoting a text) | 남친은 히어로물이라더니 | '남친이 슈퍼히어로 / 드라마랬는데'
- "I am the way into the city of woe." (read from a gate) | 나는 슬픔의 도시로 가는 길이다 | '나는 슬픔의 나라로 / 들어가는 문'

### 8.4 Subtitle readable plot text as its own beat-aligned cue, with kind-appropriate register and quotes.
Time and place cards, signs, screens, documents, chat messages and gag captions, placed on their visual beat; shorten overlapping dialogue when needed so both remain readable.
- **Screen messages (`[message]`) & personal notes (`[note]`)**: Translate each bubble/note as its own cue using the sender→addressee speech level from the REGISTERS table (e.g. teen/subordinate texting mentor/adult = 해요체 `'해피, 그냥 보내봐요 / 학교는 2시 45분에 끝나요'`, `'다음 임무 준비됐어요!'`, `'저 피터예요'`, `'피터 파커'`; close aunt/friend note = 해체). Wrap single-party messages, notes, signs, and headlines in straight single quotes `'...'`; for two-party SMS on one screen, tag the replying sender (`제발 서두르세요 / 제임스: 지금 갑니다`). Never drop clock times, numbers, or trailing signatures (`스파이더맨`, `피터 파커`).
- **Signs, close-up props, banners, headlines, captions & gag credits (`[sign]`, `[ui]`, `[caption]`, `[gag_credit]`, `[dedication]`)**: Place/date cards in Korean country-first order (`IXTENCO, MEXICO` → `멕시코, 익스텡코`, `VENICE, ITALY` → `이탈리아, 베니스`, `PRAGUE, CZECH REPUBLIC` → `체코, 프라하`, `BERLIN, GERMANY` → `독일, 베를린`, `EASTERN ALPS, AUSTRIA` → `오스트리아, 알프스 동부`, `DORSET COAST, UNITED KINGDOM` → `영국, 도싯 해안`, `NEWARK, NEW JERSEY` → `뉴저지, 뉴어크`, `SIX YEARS LATER` → `'6년 후'`); character lower-thirds as `이름 / 직함` (`'웨이드 / 드라이브맥스 영업사원'`); close-up prop inserts as concise quoted labels (`'롭 L.'`, `'인류 최고 섹시남!'`, `'라이펠드 가 / 니시에사 가'`, `'페이지 피자'`, `'액화 산소'`, `'소독용 알코올'`, `'손 세정제'`, `'내 집 / 내 규칙 / 내 커피!!'`); satirical gag credits with their punchy comedic register (`STARRING GOD'S PERFECT IDIOT` → `출연 / 신이 내린 또라이`, `SOME DOUCHEBAG'S FILM` → `제작 / 개허접 필름`).
Why: a Korean viewer can't read the English text on screen, and the plot, character action, or joke depends on it.
Not when: out-of-focus background decoration, studio logos, real opening or closing cast/crew credit crawls, or a repeat with nothing new.
- JANUARY 10, 1952 NEW JERSEY | 1952년 1월 10일 뉴저지 | 1952년 1월 10일 / 뉴저지
- ROME, ITALY (caption) | 로마, 이탈리아 | 이탈리아, 로마
- 8 YEARS LATER (time card) | (no subtitle) | '8년 후'
### 8.5 Numbers: digits, thousand separators, 만/억 for round large amounts.
Write 1,000–9,999 as 3,000 (never 3천). Round amounts of 10,000 and up: digits plus 만/억 (10만 명, 1,400만 달러, 1억 5천만 달러). Full digits only for exact figures read off something (168,571달러). English number words become digits.
Why: digits read fastest; a unit right after the number saves a conversion.
Not when: counting aloud or an age in speech (열!, 열일곱 살).
- over 1,500 tons | 천오백 톤 이상 | 1,500톤 이상
- about 3,000 rounds / 14 million dollars | 대략 3천 발 / 1400만 달러 | 3,000발쯤 / 1,400만 달러

### 8.6 Convert casual imperial measures to rounded metric; keep miles and technical specs.
Body size, weight, heights and short distances in conversation become rounded metric (three feet → 1미터, a 290-pound man → 130kg, a thousand yards → 1km). Miles (80마일) and gun or engineering specs stay.

### 8.7 Commas only after a one-word reply, a vocative or a repetition; never between clauses.
그래, / 아니, / 잠깐, / 안 돼, 안 돼! are fine. Two clauses or sentences on one line are separated by a line break or a space, never a comma or a period.
Why: a comma at a clause break is English punctuation; the line break or the space already marks the pause.
- You got a nice face. You're just a kid. | 얼굴 반반하네, 그냥 애잖아 | 넌 얼굴도 잘생겼고 / 나이도 어리고
- I got to get to class, but, um... | 수업 가야 하는데, 저기... | 이제 수업 가야 돼

### 8.8 "!" only for a real shout.
Drop "!" on routine "Let's go!", tactical callouts and flat insults; quiet dramas use it rarely. No final period ever; "?" follows real questions.

## Self-check before returning a line

- Same act (jab, plea, joke, threat, retort) as the English here? Nothing invented; no plot fact, punch word or setup lost?
- Shorter than a faithful version even when there's time? Openers, hedges, tags, courtesy formulas, repeats, tag vocatives and pure vocalizations dropped?
- Speech level right for this pair, this world and this moment: no generic 해요체, 하십시오체 only for hierarchy or public speech, archaic endings only for old-world characters, no invented 아저씨/이봐?
- No reflex 세상에/맙소사, 대체, 아,/어,; -잖아 only for shared facts, -거든 sparingly, 하지만 only for a weighty pivot?
- Profanity at the video's and speaker's ceiling and the scene's heat; dropped tokens moved into crude verbs, imperatives or tone; lone "Fuck!" = 젠장/썅; signature swears fixed; 놈/새끼 only for contempt?
- Joke or reference lands for a Korean viewer; real names kept, obscure ones turned into categories; sound jokes rebuilt or turned into the jab; labels consistent?
- Every line 14 characters or fewer, a one-line cue 12 or fewer, at most two lines, broken by sense, no comma or period at line end; "..." only trailing on a real cut-off; dashes on both lines only for two real speakers; titles and read-aloud text in '...', on-screen text plain; numbers in digits, 만/억 for round big amounts?
- Read aloud as the character: would a Korean actor say this? If not, rewrite it.
