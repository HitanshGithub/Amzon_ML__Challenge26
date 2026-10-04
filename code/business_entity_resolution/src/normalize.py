"""Text normalisation for business names and addresses.

Produces, per record:
  name_norm  - ascii, lowercase, punctuation-free, legal suffixes canonicalised
  name_core  - name_norm without legal-form / filler tokens
  name_key   - name_core with repeated letters collapsed (robust to transliteration: 'raam' -> 'ram')
  name_sq    - name_key without spaces (matches domain-style names 'coastaltungsten.com')
  legal      - sorted canonical legal tokens ('llc', 'pvt ltd', ...)
  addr_norm  - ascii, lowercase, abbreviations expanded, state codes expanded
  nums       - space-joined digit runs of the address
  flags      - is_domain, non_ascii, addr_empty

Country is used only to pick an optional state-abbreviation table; unknown countries
(e.g. France, which never appears in training) go through exactly the same path.
"""
import re
import unicodedata

import numpy as np
import pandas as pd
from unidecode import unidecode

LEGAL = {
    "private": "pvt", "pvt": "pvt", "pte": "pvt", "pvtltd": "pvt ltd",
    "limited": "ltd", "ltd": "ltd", "ltda": "ltd",
    "corporation": "corp", "corp": "corp",
    "incorporated": "inc", "inc": "inc", "lnc": "inc",
    "company": "co", "co": "co", "cos": "co",
    "llc": "llc", "llp": "llp", "lp": "lp", "plc": "plc", "pllc": "pllc", "pc": "pc", "opc": "opc",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "sci": "sci", "eurl": "eurl", "ei": "ei",
    "snc": "snc", "scop": "scop", "gmbh": "gmbh", "ag": "ag", "bv": "bv", "nv": "nv",
}
# honorifics and connector words the noise process adds (share of occurrences only on the
# S2/S3 side of true pairs: sri/shri/smt/mr/dr ~100%, the ~92%)
FILLER = {"the", "and", "of", "et", "de", "la", "le", "les", "du", "des", "ms",
          "sri", "shri", "shree", "smt", "mr", "mrs", "dr", "id"}
# word abbreviations observed between S1 and S2/S3 names in training pairs; both sides are
# mapped to the long form so exact-token comparisons line up
NAME_ABBR = {
    "bros": "brothers", "bro": "brothers", "tech": "technologies", "technology": "technologies",
    "techs": "technologies", "intl": "international", "svc": "services", "svcs": "services",
    "service": "services", "mgmt": "management", "mfg": "manufacturing", "assoc": "associates",
    "assocs": "associates", "univ": "university", "inst": "institute", "natl": "national",
    "hosp": "hospital", "ent": "enterprises", "entp": "enterprises", "enterprise": "enterprises",
    "sys": "systems", "syst": "systems", "system": "systems", "phys": "physicians",
    "solution": "solutions", "grp": "group", "ctr": "center", "centre": "center",
    "consultant": "consultants", "holding": "holdings", "venture": "ventures", "product": "products",
}
# "<brand> DBA <real name>": the real business name is the part after the marker
_ALIAS = re.compile(r"\s+(?:d\s*/\s*b\s*/\s*a|dba|a\s*/\s*k\s*/\s*a|aka|f\s*/\s*k\s*/\s*a|fka|t\s*/\s*a|nee|"
                    r"formerly(?:\s+known\s+as)?\s*:?|doing\s+business\s+as)\s+")
_ID_TAG = re.compile(r"\(\s*id\s*:?\s*\d+\s*\)")
_ZERO_WIDTH = re.compile(r"[​-‏⁠﻿­]")
ORDINALS = {"first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th", "fifth": "5th",
            "sixth": "6th", "seventh": "7th", "eighth": "8th", "ninth": "9th", "tenth": "10th"}

ADDR_ABBR = {
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "bd": "boulevard", "bvd": "boulevard", "dr": "drive", "ln": "lane",
    "ct": "court", "cir": "circle", "pl": "place", "pkwy": "parkway", "hwy": "highway",
    "sq": "square", "ter": "terrace", "trl": "trail", "fwy": "freeway", "expy": "expressway",
    "mt": "mount", "ft": "fort", "rte": "route", "r": "rue", "chem": "chemin", "imp": "impasse",
    "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
    "opp": "opposite", "nr": "near", "mkt": "market", "ngr": "nagar", "clny": "colony",
    "dist": "district", "distt": "district", "tq": "taluk", "tal": "taluk",
    "cv": "cove", "tpke": "turnpike", "rdg": "ridge", "pt": "point", "plt": "plot",
    # frequent typos seen in training pairs
    "rod": "road", "rad": "road", "stret": "street", "flor": "floor", "drve": "drive", "ciy": "city",
    **ORDINALS,
}
ADDR_DROP = {"no", "h", "hno", "house", "door", "number", "unit", "suite", "ste", "apt",
             "apartment", "fl", "floor", "city", "po", "post", "null", "none", "na", "ndeg"}

US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california",
    "co": "colorado", "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia",
    "hi": "hawaii", "id": "idaho", "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas",
    "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland", "ma": "massachusetts",
    "mi": "michigan", "mn": "minnesota", "ms": "mississippi", "mo": "missouri", "mt": "montana",
    "ne": "nebraska", "nv": "nevada", "nh": "new hampshire", "nj": "new jersey",
    "nm": "new mexico", "ny": "new york", "nc": "north carolina", "nd": "north dakota",
    "oh": "ohio", "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania", "ri": "rhode island",
    "sc": "south carolina", "sd": "south dakota", "tn": "tennessee", "tx": "texas", "ut": "utah",
    "vt": "vermont", "va": "virginia", "wa": "washington", "wv": "west virginia",
    "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia", "pr": "puerto rico",
}
IN_STATES = {
    "dl": "delhi", "hr": "haryana", "up": "uttar pradesh", "mh": "maharashtra", "ka": "karnataka",
    "tn": "tamil nadu", "wb": "west bengal", "gj": "gujarat", "rj": "rajasthan",
    "mp": "madhya pradesh", "pb": "punjab", "ap": "andhra pradesh", "ts": "telangana",
    "tg": "telangana", "kl": "kerala", "br": "bihar", "or": "odisha", "od": "odisha",
    "jh": "jharkhand", "cg": "chhattisgarh", "ct": "chhattisgarh", "uk": "uttarakhand",
    "ut": "uttarakhand", "hp": "himachal pradesh", "jk": "jammu and kashmir", "ga": "goa",
    "as": "assam", "ch": "chandigarh", "py": "puducherry", "mn": "manipur", "ml": "meghalaya",
    "tr": "tripura", "nl": "nagaland", "ar": "arunachal pradesh", "mz": "mizoram", "sk": "sikkim",
}
STATE_TABLES = {"us": US_STATES, "india": IN_STATES}

# France (test-only country). Applied only to records whose country is France, so US/India
# normalisation - and every model trained on it - is unchanged.
# * Region / department names are whole comma components; S1 uses the region, S2/S3 often the
#   department, so both are dropped rather than compared.
# * "St"/"Ste" mean Saint/Sainte in French addresses (St-Nazaire, St.-Herblain), not Street.
FR_ADDR_ABBR = {"st": "saint", "ste": "sainte", "all": "allee", "ch": "chemin", "chem": "chemin",
                "crs": "cours", "fg": "faubourg", "fbg": "faubourg", "qu": "quai", "sq": "square",
                "bd": "boulevard", "blvd": "boulevard", "av": "avenue", "ave": "avenue", "r": "rue",
                "pl": "place", "imp": "impasse", "rte": "route", "rdc": "rez de chaussee"}
FR_ADDR_DROP_COMPONENTS = {"hauts de france", "nouvelle aquitaine", "pays de la loire",
                           "nord", "gironde", "loire atlantique", "pas de calais", "france"}
FR_NAME_ABBR = {"cie": "compagnie", "ste": "societe", "ets": "etablissements", "etbs": "etablissements",
                "assoc": "association", "asso": "association", "fr": "freres"}
COUNTRY_ADDR_ABBR = {"france": FR_ADDR_ABBR}
COUNTRY_DROP_COMPONENTS = {"france": FR_ADDR_DROP_COMPONENTS}
COUNTRY_NAME_ABBR = {"france": FR_NAME_ABBR}

_DOMAIN = re.compile(r"^\s*(?:www\.)?([a-z0-9\-]+)\.(?:com|in|net|org|co|co\.in|biz|info|fr|io|us)\s*$")
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b", "@": "a", "$": "s"})
_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
_SPACES = re.compile(r"\s+")
_REPEAT = re.compile(r"(.)\1+")
_DIGITS = re.compile(r"\d+")


# Non-Latin token -> English token dictionary learned from training pairs (see data.learn_translit).
TRANSLIT: dict = {}


def set_translit(d: dict):
    global TRANSLIT
    TRANSLIT = d


# A "word" is any run of characters that are neither whitespace nor ASCII punctuation.
# (Python's \w does not cover Indic vowel signs, which would split words mid-way.)
WORD = re.compile(r"[^\s!-/:-@\[-`{-~]+")


def words(s: str):
    return WORD.findall(s.lower())


def apply_translit(raw: str) -> str:
    return WORD.sub(lambda m: TRANSLIT.get(m.group(0).lower(), m.group(0)), raw)


def to_ascii(s: str) -> str:
    # unidecode handles Devanagari ('राम' -> 'raam') and accents ('é' -> 'e').
    return unidecode(unicodedata.normalize("NFKC", s)).lower()


def _deleet(tok: str) -> str:
    letters = sum(c.isalpha() for c in tok)
    digits = sum(c.isdigit() for c in tok)
    if letters >= 3 and 0 < digits <= 2:
        return tok.translate(_LEET)
    return tok


def norm_name(raw: str, country: str = ""):
    extra_abbr = COUNTRY_NAME_ABBR.get(country.strip().lower(), {})
    raw = _ZERO_WIDTH.sub("", raw)
    non_ascii = any(ord(c) > 127 for c in raw)
    if non_ascii and TRANSLIT:
        raw = apply_translit(raw)
    s = to_ascii(raw)
    s = _ID_TAG.sub(" ", s)                                 # "(ID: 93967)"
    parts = _ALIAS.split(s)
    has_alias = len(parts) > 1
    if has_alias:                                          # "Avixylo dba Gonzalez Landscaping" -> real name last
        s = parts[-1]
    s = re.sub(r"^\s*m\s*/\s*s\b\.?", " ", s)        # "M/s" prefix used in India
    m = _DOMAIN.match(s)
    is_domain = m is not None
    if is_domain:
        s = m.group(1).replace("-", " ")
    s = s.replace("&", " and ").replace("+", " plus ")
    s = s.replace(".", "")                                 # S.A.S -> sas, Pvt. -> pvt
    s = _NON_ALNUM.sub(" ", s)
    toks = [_deleet(t) for t in s.split()]
    norm, core, legal = [], [], []
    for t in toks:
        canon = LEGAL.get(t) or LEGAL.get(_REPEAT.sub(r"\1", t))
        if canon:
            norm.append(canon)
            legal.extend(canon.split())
        else:
            t = extra_abbr.get(t, NAME_ABBR.get(t, t))
            norm.append(t)
            if t not in FILLER:
                core.append(t)
    if not core:                                           # name was only legal words
        core = [t for t in norm]
    name_norm = " ".join(norm)
    name_core = " ".join(core)
    name_key = _REPEAT.sub(r"\1", name_core)
    return (name_norm, name_core, name_key, name_key.replace(" ", ""), " ".join(sorted(set(legal))),
            is_domain, non_ascii, has_alias)


def norm_addr(raw: str, country: str):
    if not raw:
        return "", ""
    ckey = country.strip().lower()
    states = STATE_TABLES.get(ckey, {})
    abbr = {**ADDR_ABBR, **COUNTRY_ADDR_ABBR.get(ckey, {})}
    drop_comp = COUNTRY_DROP_COMPONENTS.get(ckey, set())
    if TRANSLIT and any(ord(c) > 127 for c in raw):
        raw = ", ".join(apply_translit(c.strip()) for c in raw.split(","))
    s = to_ascii(_ZERO_WIDTH.sub("", raw))
    s = re.sub(r"\bn\s*/\s*a\b", " ", s).replace(".", "")
    parts = []
    for comp in s.split(","):
        c = _SPACES.sub(" ", _NON_ALNUM.sub(" ", comp)).strip()
        if c in drop_comp:
            continue
        if c in states:
            c = states[c]
        parts.append(c)
    toks = []
    for t in " ".join(parts).split():
        if t in ADDR_DROP:
            continue
        if t.isdigit():
            t = t.lstrip("0") or "0"                       # zero-padded numbers: 00412 -> 412
        toks.append(abbr.get(t, t))
    addr = " ".join(toks)
    nums = " ".join(d.lstrip("0") or "0" for d in _DIGITS.findall(addr))
    return addr, nums


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    names = [norm_name(x, c) for x, c in zip(df["business_name"].tolist(), df["country"].tolist())]
    addrs = [norm_addr(a, c) for a, c in zip(df["business_address"].tolist(), df["country"].tolist())]
    out = pd.DataFrame({
        "entity_id": df["entity_id"].values,
        "country": df["country"].str.strip().str.lower().values,
    })
    cols = ["name_norm", "name_core", "name_key", "name_sq", "legal", "is_domain", "non_ascii", "has_alias"]
    for i, c in enumerate(cols):
        out[c] = [n[i] for n in names]
    out["addr_norm"] = [a[0] for a in addrs]
    out["nums"] = [a[1] for a in addrs]
    out["addr_empty"] = (out["addr_norm"] == "").astype(np.int8)
    for c in ("is_domain", "non_ascii", "has_alias"):
        out[c] = out[c].astype(np.int8)
    return out


def normalize_parallel(df: pd.DataFrame, n_jobs: int) -> pd.DataFrame:
    if n_jobs <= 1 or len(df) < 50_000:
        return normalize_frame(df)
    from multiprocessing import Pool
    chunks = np.array_split(np.arange(len(df)), n_jobs * 4)
    with Pool(n_jobs, initializer=set_translit, initargs=(TRANSLIT,)) as pool:
        parts = pool.map(normalize_frame, [df.iloc[c] for c in chunks])
    return pd.concat(parts, ignore_index=True)
