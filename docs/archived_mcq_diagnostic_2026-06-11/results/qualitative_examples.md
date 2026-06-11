# Qualitative Examples

## Raw wrong, proposed right

### fixed_q020 / proposed

- Gold: `B`
- Parsed: `B`
- Raw response: `"It"" avoids ""delay, keeps the progra""m consistent"`
- Rewritten question: Consider a method that packs a program with the surrounding files it expects. A decision-maker is comparing methods for running a program on different machines. Why might this method be selected? Explain the functional reason in ordinary terms and choose the best option.

### fixed_q043 / proposed

- Gold: `A`
- Parsed: `A`
- Raw response: `"It u""ses socia""l links to pass mess""ages rapidly"`
- Rewritten question: Consider a method that lets many people publish short messages to connected groups. When attempting spreading a public notice quickly, what is the clearest justification for using this method? Explain the functional reason in ordinary terms and choose the best option.

### fixed_q055 / proposed

- Gold: `A`
- Parsed: `A`
- Raw response: `"It ""places ""more ""likely items before"" less likely"`
- Rewritten question: Consider a method that orders messages by calculated signals of likely interest. In a discussion about choosing which notices a reader sees first, which account of this method is the most accurate? Explain the functional reason in ordinary terms and choose the best option.

## Raw right, proposed wrong

### fixed_q031 / proposed

- Gold: `A`
- Parsed: `INVALID`
- Raw response: `"A ""matter""s most, because it fo""rms pictures"`
- Rewritten question: Consider a method that sends high-frequency sound and measures echoes. A group is evaluating tools for watching a moving organ inside the body. Which property of this method matters most? Explain the functional reason in ordinary terms and choose the best option.

### fixed_q067 / proposed

- Gold: `A`
- Parsed: `C`
- Raw response: `"It h""elps tire""s keep grip f""or steering."`
- Rewritten question: Consider a method that embeds strong fibers inside a binding material. Which mechanism explains why this method can help with making a strong light vehicle part? Explain the functional reason in ordinary terms and choose the best option.

### fixed_q075 / proposed

- Gold: `A`
- Parsed: `B`
- Raw response: `"It fo""cuses atte""ntion on smal""l particles,"`
- Rewritten question: Consider a method that uses many wind-driven rotors connected to generators. When attempting generating electricity on open land, what is the clearest justification for using this method? Explain the functional reason in ordinary terms and choose the best option.

## Invalid responses

### fixed_q001 / raw

- Gold: `C`
- Parsed: `INVALID`
- Raw response: `"C c""hooses best."`
- Rewritten question: A team is working on reducing unsupported answers from an automatic writing system. Which explanation best supports using RAG for LLM hallucination?

### fixed_q003 / raw

- Gold: `A`
- Parsed: `INVALID`
- Raw response: `"It helps"" to keep the two systems"" cooperating"`
- Rewritten question: Which mechanism explains why API can help with letting two software services cooperate?

### fixed_q003 / proposed

- Gold: `A`
- Parsed: `INVALID`
- Raw response: `"It fulfils ""all choices."`
- Rewritten question: Consider a method that offers a published set of commands that one machine can use to request work from another. Which mechanism explains why this method can help with letting two software services cooperate? Explain the functional reason in ordinary terms and choose the best option.

## Repaired rewrites

No examples.

## High leakage or choice-copying

No examples.
