# Accidental R2 — KeySignature/Local Separation Acceptance v1

Status: **DESIGN-FROZEN / SHADOW-ONLY**

This acceptance contract separates visual accidental-symbol evidence from deterministic musical context resolution.

## Safety boundary

- No sealed TEST access.
- No optimizer or training changes.
- No checkpoint mutation.
- No Resolver wiring or production promotion.
- No key/mode guess may be forced from ambiguous evidence.
- Deterministic components must fail closed as `AMBIGUOUS`, `AMBIGUOUS_TARGET`, or `REJECTED`.

## Specialist evidence contract

The visual specialist layer reports only observable evidence:

- `symbol`: `sharp | flat | natural | double_sharp | double_flat`
- `bbox`
- `center`
- `confidence`
- `staff_id`

It must not decide whether the symbol is a key signature or a local accidental, and it must not infer a major/minor key.

## Deterministic context router

The context router emits exactly one of:

- `KEY_SIGNATURE`
- `LOCAL_ACCIDENTAL`
- `AMBIGUOUS`
- `AMBIGUOUS_TARGET`
- `REJECTED`

Key-signature routing must use more than x-position alone. It must consider clef-relative key-signature zone, staff position, canonical symbol order, and structural boundaries before the first note/time-signature region.

## Canonical key-signature order

Sharps: `F C G D A E B`

Flats: `B E A D G C F`

The AI does not need to learn these musical rules. They belong to the deterministic Key Signature Composer.

## Core acceptance cases

| ID | Visual situation | Specialist evidence | Expected context | Expected deterministic result |
|---|---|---|---|---|
| K01 | One `#` after clef in key-signature zone | 1 x sharp | KEY_SIGNATURE | `fifths=+1`; F#; possible keys G major / E minor; `mode=UNKNOWN` |
| K02 | Two separate `# #` after clef | 2 x sharp | KEY_SIGNATURE | `fifths=+2`; F#, C#; possible keys D major / B minor; `mode=UNKNOWN` |
| K03 | Three separate `# # #` after clef | 3 x sharp | KEY_SIGNATURE | `fifths=+3`; F#, C#, G#; possible keys A major / F# minor; `mode=UNKNOWN` |
| K04 | One `b` after clef in key-signature zone | 1 x flat | KEY_SIGNATURE | `fifths=-1`; Bb; possible keys F major / D minor; `mode=UNKNOWN` |
| K05 | Two separate `b b` after clef | 2 x flat | KEY_SIGNATURE | `fifths=-2`; Bb, Eb; possible keys Bb major / G minor; `mode=UNKNOWN` |
| K06 | Three separate `b b b` after clef | 3 x flat | KEY_SIGNATURE | `fifths=-3`; Bb, Eb, Ab; possible keys Eb major / C minor; `mode=UNKNOWN` |
| L01 | `#` immediately left of a note inside a measure | sharp | LOCAL_ACCIDENTAL | bind to target note; `alter=+1` |
| L02 | `b` immediately left of a note inside a measure | flat | LOCAL_ACCIDENTAL | bind to target note; `alter=-1` |
| L03 | natural immediately left of a note inside a measure | natural | LOCAL_ACCIDENTAL | bind to target note; `alter=0` |
| L04 | one double-sharp glyph inside a measure | double_sharp | LOCAL_ACCIDENTAL | one target note; `alter=+2`; must not be counted as two sharps |
| L05 | one double-flat glyph inside a measure | double_flat | LOCAL_ACCIDENTAL | one target note; `alter=-2`; must not be counted as two flats |
| L06 | 1-sharp key signature plus local natural on F | sharp + natural | KEY_SIGNATURE + LOCAL_ACCIDENTAL | retain F# key signature; local F becomes F-natural for applicable scope |
| L07 | 2-flat key signature plus local sharp | flats + sharp | KEY_SIGNATURE + LOCAL_ACCIDENTAL | retain Bb/Eb key signature; target note gets `alter=+1` |
| L08 | 2-sharp key signature plus local flat | sharps + flat | KEY_SIGNATURE + LOCAL_ACCIDENTAL | retain F#/C# key signature; target note gets `alter=-1` |
| L09 | Two separate local sharps attached to two different notes | 2 x sharp | LOCAL_ACCIDENTAL | two independent local alterations; must not become key signature |
| L10 | One double-sharp glyph next to one note | double_sharp | LOCAL_ACCIDENTAL | one symbol, one target, `alter=+2` |
| A01 | Sharp near staff start but geometrically compatible with both key zone and first-note local accidental | sharp | AMBIGUOUS | no forced key signature or local binding |
| A02 | Two sharps in key zone but staff positions/order violate canonical key-signature sequence | 2 x sharp | AMBIGUOUS | must not force D major / B minor |
| A03 | Natural in key-signature region without sufficient cancellation/key-change context | natural | AMBIGUOUS | no forced interpretation |
| A04 | Symbol bbox overlaps two staffs with no unique staff assignment | any accidental | AMBIGUOUS | no staff binding |
| A05 | Local accidental is equally plausible for two noteheads | any accidental | AMBIGUOUS_TARGET | no note mutation |
| A06 | Non-finite confidence/coordinates or invalid bbox | any accidental | REJECTED | fail closed |

## Key-signature versus double-accidental invariant

Two separate sharps `# #` are not the same object as one double-sharp glyph. Two separate flats `b b` are not the same object as one double-flat glyph.

Examples:

- `# #` in the valid key-signature zone and canonical staff positions -> `fifths=+2`.
- one double-sharp glyph next to a note -> local accidental `alter=+2`.
- `b b` in the valid key-signature zone and canonical staff positions -> `fifths=-2`.
- one double-flat glyph next to a note -> local accidental `alter=-2`.

## Mode inference rule

Key signature alone must not force major versus minor.

Examples:

- `fifths=+1` -> possible keys G major / E minor; `mode=UNKNOWN`.
- `fifths=+2` -> possible keys D major / B minor; `mode=UNKNOWN`.
- `fifths=-1` -> possible keys F major / D minor; `mode=UNKNOWN`.
- `fifths=-2` -> possible keys Bb major / G minor; `mode=UNKNOWN`.

Major/minor resolution belongs to a later musical-context validator/composer.

## Local accidental scope rule

The Local Accidental Resolver applies an accepted accidental to its deterministic target and scope. At minimum, the acceptance contract requires measure-local behavior and barline reset back to the active key signature. Tie, octave, voice, key-change, cautionary and cancellation edge cases belong to a later dedicated deterministic acceptance pack.

## Acceptance gate

For deterministic context/composer behavior, expected-case correctness is **100%** on this acceptance pack.

Any case that cannot be resolved uniquely must fail closed rather than mutate musical meaning.

This contract is shadow-only and does not authorize Resolver wiring, production promotion, TEST opening, checkpoint changes, or merge.
