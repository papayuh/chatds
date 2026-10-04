"""KB1-KB3 reader: seek+read only, like the DS. Sector reads are counted in `sector_reads`."""
import heapq
import re, struct, unicodedata

SECTOR = 512
sector_reads = 0

# key rules (mirror in C: lowercase, non [a-z0-9] runs -> one space, trim; FNV-1a 32)
_MAP = str.maketrans({"–": "-", "—": "-", "−": "-", "‘": "'", "’": "'",
                      "“": '"', "”": '"', "…": "...", " ": " ", "×": "x"})

def ascii_only(s):
    s = unicodedata.normalize("NFKD", s.translate(_MAP))
    return s.encode("ascii", "ignore").decode()

def normalize(s):
    return re.sub(r"[^a-z0-9]+", " ", ascii_only(s).lower()).strip()

def fnv1a(b):
    h = 2166136261
    for c in b:
        h = ((h ^ c) * 16777619) & 0xFFFFFFFF
    return h


def variants(w):
    """The word itself plus every single-character deletion (SymSpell delete-1)."""
    return {w} | {w[:i] + w[i + 1:] for i in range(len(w))}


_last = None  # one-sector buffer, like the DS: re-reading the buffered sector is free

def _read(f, off, n):
    global sector_reads, _last
    for s in range(off // SECTOR, (off + n - 1) // SECTOR + 1):
        if s != _last:
            sector_reads += 1
            _last = s
    f.seek(off)
    return f.read(n)

def lookup(path, query_key):
    key = normalize(query_key).encode("ascii")
    if not key or len(key) > 63:
        return None
    h = fnv1a(key)
    with open(path, "rb") as f:
        hd = f.read(SECTOR)  # header: loaded once at mount on the DS, not counted
        _, _, nb, nk, nrec, bt, ko, to, size = struct.unpack("<4sIIIIIIII", hd[:36])
        b = h & (nb - 1)
        a, e = struct.unpack("<2I", _read(f, bt + 4 * b, 8))
        blk = _read(f, ko + a, e - a) if e > a else b""
        i = 0
        while i < len(blk):
            eh, kl = struct.unpack_from("<IB", blk, i)
            if eh == h and blk[i + 5:i + 5 + kl] == key:
                (t,) = struct.unpack_from("<I", blk, i + 5 + kl)
                n = _read(f, to + t, 1)[0]
                return _read(f, to + t + 1, n).decode("ascii")
            i += 9 + kl
    return None


def fuzzy_words(path, q, max_n=4):
    """Indexed words within one edit (or one transposition) of q, most frequent first.
    Probes len(q)+1 delete-variants of q against the KB2 fuzzy section."""
    qv = variants(q)
    with open(path, "rb") as f:
        hd = f.read(SECTOR)
        fnb, fbt, fent, fwo, _, _ = struct.unpack_from("<6I", hd, 36)
        offs = set()
        for v in qv:
            h = fnv1a(v.encode("ascii"))
            a, e = struct.unpack("<2I", _read(f, fbt + 4 * (h & (fnb - 1)), 8))
            blk = _read(f, fent + a, e - a) if e > a else b""
            for i in range(0, len(blk), 8):
                eh, wo = struct.unpack_from("<II", blk, i)
                if eh == h:
                    offs.add(wo)
        out = []
        for wo in heapq.nsmallest(16, offs):  # C twin keeps 16 (NOFF)
            w = _read(f, fwo + wo + 1, _read(f, fwo + wo, 1)[0]).decode("ascii")
            if w != q and variants(w) & qv:  # hash hit -> real edit relation
                out.append(w)
                if len(out) == max_n:
                    break
        return out
