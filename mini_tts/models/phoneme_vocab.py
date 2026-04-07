"""
Phoneme vocabulary for mini_tts.

Covers the full IPA set used in English (ARPAbet-derived + stress markers).
All text preprocessing is done offline (phonemizer library) — this module
only handles integer ID mapping at inference time.

Usage:
    vocab = PhonemeVocab()
    ids   = vocab.encode(["AH", "L", "OW"])   # → [5, 17, 29]
    back  = vocab.decode(ids)                  # → ["AH", "L", "OW"]
"""

from __future__ import annotations
from typing import List, Dict, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Special tokens
# ─────────────────────────────────────────────────────────────────────────────
PAD   = "<pad>"
UNK   = "<unk>"
BOS   = "<bos>"   # beginning of sentence
EOS   = "<eos>"   # end of sentence
SIL   = "<sil>"   # silence / breath
SPACE = "<sp>"    # word boundary

SPECIAL_TOKENS = [PAD, UNK, BOS, EOS, SIL, SPACE]

# ─────────────────────────────────────────────────────────────────────────────
# ARPAbet phoneme set (covers American English completely)
# Vowels + diphthongs (with stress variants 0/1/2)
# ─────────────────────────────────────────────────────────────────────────────
_ARPABET_VOWELS = [
    "AA", "AE", "AH", "AO", "AW", "AY",
    "EH", "ER", "EY",
    "IH", "IY",
    "OW", "OY",
    "UH", "UW",
]

_ARPABET_CONSONANTS = [
    "B",  "CH", "D",  "DH", "F",  "G",
    "HH", "JH", "K",  "L",  "M",  "N",
    "NG", "P",  "R",  "S",  "SH", "T",
    "TH", "V",  "W",  "Y",  "Z",  "ZH",
]

# Stress-marked versions of vowels (0=no stress, 1=primary, 2=secondary)
_STRESSED_VOWELS: List[str] = []
for v in _ARPABET_VOWELS:
    for stress in ("0", "1", "2"):
        _STRESSED_VOWELS.append(f"{v}{stress}")

# Additional symbols used by espeak / phonemizer (IPA mode)
_IPA_EXTRAS = [
    "ə",  "ɪ",  "ʊ",  "ɛ",  "æ",  "ɑ",  "ɔ",
    "ʌ",  "ɜ",  "i",  "u",  "e",  "o",  "a",
    "p",  "b",  "t",  "d",  "k",  "g",  "ʔ",
    "m",  "n",  "ŋ",  "f",  "v",  "θ",  "ð",
    "s",  "z",  "ʃ",  "ʒ",  "h",  "tʃ", "dʒ",
    "l",  "r",  "w",  "j",
    "ˈ",  "ˌ",  ".",  ",",  "!",  "?",  "-",
]

# Full phoneme list = specials + ARPAbet (unstressed) + stressed vowels + consonants + IPA extras
_ALL_PHONEMES = (
    SPECIAL_TOKENS
    + _ARPABET_VOWELS
    + _STRESSED_VOWELS
    + _ARPABET_CONSONANTS
    + _IPA_EXTRAS
)

# De-duplicate while preserving order
_seen: set = set()
PHONEME_LIST: List[str] = []
for _p in _ALL_PHONEMES:
    if _p not in _seen:
        PHONEME_LIST.append(_p)
        _seen.add(_p)

VOCAB_SIZE = len(PHONEME_LIST)   # ~120–140

PAD_ID   = PHONEME_LIST.index(PAD)
UNK_ID   = PHONEME_LIST.index(UNK)
BOS_ID   = PHONEME_LIST.index(BOS)
EOS_ID   = PHONEME_LIST.index(EOS)
SIL_ID   = PHONEME_LIST.index(SIL)
SPACE_ID = PHONEME_LIST.index(SPACE)


# ─────────────────────────────────────────────────────────────────────────────
# PhonemeVocab
# ─────────────────────────────────────────────────────────────────────────────

class PhonemeVocab:
    """
    Integer ↔ phoneme mapping.

    Thread-safe (read-only after construction).
    All lookups are O(1) dict operations.
    """

    def __init__(self) -> None:
        self._phonemes: List[str]     = PHONEME_LIST
        self._p2i:      Dict[str, int] = {p: i for i, p in enumerate(PHONEME_LIST)}

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        return len(self._phonemes)

    @property
    def pad_id(self)   -> int: return PAD_ID
    @property
    def unk_id(self)   -> int: return UNK_ID
    @property
    def bos_id(self)   -> int: return BOS_ID
    @property
    def eos_id(self)   -> int: return EOS_ID
    @property
    def sil_id(self)   -> int: return SIL_ID
    @property
    def space_id(self) -> int: return SPACE_ID

    # ── Encoding / decoding ───────────────────────────────────────────────────

    def encode(self, phonemes: List[str], add_bos: bool = False,
               add_eos: bool = False) -> List[int]:
        """Convert a list of phoneme strings to integer IDs."""
        ids = [self._p2i.get(p, UNK_ID) for p in phonemes]
        if add_bos:
            ids = [BOS_ID] + ids
        if add_eos:
            ids = ids + [EOS_ID]
        return ids

    def decode(self, ids: List[int]) -> List[str]:
        """Convert integer IDs back to phoneme strings."""
        return [self._phonemes[i] if 0 <= i < len(self._phonemes) else UNK
                for i in ids]

    def encode_sentence(self, phonemes: List[str]) -> List[int]:
        """Encode with BOS + EOS framing (used at training time)."""
        return self.encode(phonemes, add_bos=True, add_eos=True)

    # ── Utilities ─────────────────────────────────────────────────────────────

    def pad_sequence(self, id_seqs: List[List[int]]) -> Tuple:
        """
        Pad a list of id sequences to the same length.
        Returns (padded_array, lengths) as plain lists.
        Use torch.tensor() on the caller side to get tensors.
        """
        max_len = max(len(s) for s in id_seqs)
        padded  = [s + [PAD_ID] * (max_len - len(s)) for s in id_seqs]
        lengths = [len(s) for s in id_seqs]
        return padded, lengths

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return f"PhonemeVocab(size={self.vocab_size})"


# ─────────────────────────────────────────────────────────────────────────────
# Text preprocessing helpers (offline use only — requires phonemizer)
# ─────────────────────────────────────────────────────────────────────────────

def text_to_phonemes_espeak(text: str, language: str = "en-us") -> List[str]:
    """
    Convert raw text → ARPAbet / IPA phoneme list using phonemizer + espeak.

    Requires:  pip install phonemizer
               apt install espeak-ng  (or brew install espeak)

    Returns list of phoneme strings ready for PhonemeVocab.encode().
    This is an OFFLINE preprocessing step — not called at inference time.
    """
    try:
        from phonemizer import phonemize
        from phonemizer.backend import EspeakBackend
    except ImportError:
        raise ImportError(
            "phonemizer not installed. Run: pip install phonemizer\n"
            "Also install espeak-ng: apt install espeak-ng"
        )
    result = phonemize(
        text,
        backend="espeak",
        language=language,
        with_stress=True,
        preserve_punctuation=False,
        language_switch="remove-flags",
    )
    # Split on spaces and filter empty
    phonemes = [p.strip() for p in result.split() if p.strip()]
    return phonemes


def text_to_ids(text: str, vocab: PhonemeVocab, language: str = "en-us") -> List[int]:
    """Convenience: text → phonemes → IDs (offline preprocessing)."""
    phonemes = text_to_phonemes_espeak(text, language=language)
    return vocab.encode_sentence(phonemes)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    vocab = PhonemeVocab()
    print(f"Vocab size: {vocab.vocab_size}")
    print(f"PAD={vocab.pad_id}, UNK={vocab.unk_id}, BOS={vocab.bos_id}, "
          f"EOS={vocab.eos_id}, SIL={vocab.sil_id}")

    seq  = ["HH", "AH0", "L", "OW1"]
    ids  = vocab.encode(seq)
    back = vocab.decode(ids)
    print(f"Encode: {seq}")
    print(f"  → IDs:  {ids}")
    print(f"  → Back: {back}")
    assert back == seq, "Round-trip failed"
    print("✓ Round-trip OK")

    framed = vocab.encode_sentence(seq)
    assert framed[0] == vocab.bos_id
    assert framed[-1] == vocab.eos_id
    print(f"Framed: {framed}")
    print("✓ All checks passed")
