# KB3 format (little-endian; KB2 layout, precomputed context records)

KB1 = magic `"KB1\0"`, version 1, no fuzzy section. KB2 appended the fuzzy section after the text area. KB3 (current) keeps the KB2 layout byte for byte; only the magic, version and record contents change: each record is now the finished context sentence, so readers never cut text (KB2 stored up to 240 bytes and the runtime trimmed them to 80 chars, often in the middle of a phrase).

## Context sentence (what a record holds)
`build_kb.context_sentence`, at most 100 chars, always ends cleanly:
1. the first sentence, if it is <= 100 chars (kept as is);
2. else the longest prefix ending at a clause boundary (`,` `;` `:` not before a digit, or just before ` and ` / ` which ` / ` who ` / ` that `; no ` and ` cut after `between`/`both`) that is >= 30 chars with balanced quotes and parentheses;
3. else the first 99 chars cut at a word boundary, then trailing function words and prepositions dropped (`from to of the and ...`, list `_FUNC`).
Cases 2 and 3 drop trailing punctuation and end with `.`.

Built by `build_kb.py` from Simple English Wikipedia; read by `kb.py` (reference reader, seek+read only).

## Key normalization
ASCII-transliterate (NFKD, drop non-ASCII, map en/em dash, curly quotes), lowercase, every run of
non `[a-z0-9]` becomes one space, trim. Keys longer than 63 bytes are dropped.
Hash: FNV-1a 32-bit (offset 2166136261, prime 16777619) over the key bytes. `bucket = hash & (n_buckets-1)`.

## Layout
Bucket table, key area and text area each start 512-aligned (zero padded).

| area | offset | contents |
|---|---|---|
| header | 0 (512 B, zero padded) | `"KB3\0"`, u32 version=3, n_buckets (power of 2, >= n_keys/4), n_keys, n_records, bucket_table_offset, keys_offset, text_offset, file_size, then fz_n_buckets, fz_bucket_offset, fz_entries_offset, fz_words_offset, fz_n_entries, fz_n_words (6 x u32 at byte 36), then ASCII source/license string |
| bucket table | bucket_table_offset (=512) | (n_buckets+1) u32 offsets relative to keys_offset; bucket b = `[off[b], off[b+1])` |
| key area | keys_offset | per entry, sorted by bucket then hash: u32 hash, u8 keylen, key bytes, u32 text_offset (relative to text_offset) |
| text area | text_offset | records: u8 len (<=100) + ASCII bytes: the context sentence above. One record per distinct sentence, shared by an article's title and redirect keys. Parenthetical `( ... )` spans are stripped before sentence selection |
| fuzzy words | fz_words_offset (512-aligned) | every key word of 5..20 chars (not all digits), most frequent first: u8 len + ASCII bytes. A word's offset is its frequency rank |
| fuzzy buckets | fz_bucket_offset | (fz_n_buckets+1) u32 offsets into the entries area; bucket = `hash & (fz_n_buckets-1)` |
| fuzzy entries | fz_entries_offset | 8 bytes each, sorted by bucket then hash then word offset: u32 fnv1a(variant), u32 word offset (relative to fz_words_offset). Variants of a word = the word plus each single-character deletion |

`file_size` must equal the real file size (loader verifies).

## Lookup (target <= 3 sector reads)
1. read `off[b]`, `off[b+1]` (8 bytes at `bucket_table_offset + 4b`)
2. read the bucket's key bytes; compare hash, then keylen+key
3. read the length byte at `text_offset + text_off`, then the text
Header is read once at mount. Measured: about 3.3 sector reads per lookup (a bucket or text record
occasionally straddles a sector boundary).

## Fuzzy lookup (typos)
For a query word q of >= 5 chars: hash q and each of its deletions, read each bucket (one read each), collect word
offsets whose hash matches, read those words, keep words w != q that share a variant with q (this covers a
missing/extra character, a substitution and an adjacent transposition), best (lowest offset) 4 first. The runtime then
rebuilds the n-gram with the candidate in place of q and does a normal exact lookup, so a wrong candidate costs a
lookup, not a wrong answer.
