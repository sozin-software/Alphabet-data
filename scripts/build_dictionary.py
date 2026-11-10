#!/usr/bin/env python3
"""Builds Alphabet's dictionary database from free sources.

Inputs (downloaded by data/download_sources.sh):
  wiktextract-en.jsonl  English Wiktionary, extracted by Wiktextract (kaikki.org), CC BY-SA
  oewn-2025.xml.gz      Open English WordNet 2025, CC BY 4.0
  mthesaur.txt          Moby Thesaurus II, public domain

Output: one SQLite file (see SCHEMA below) that the app downloads once and reads offline.

Run:  python build_dictionary.py RAW_DIR OUT.db
Needs: pip install orjson wordfreq
"""

import gzip
import os
import re
import sqlite3
import sys
import time
import unicodedata
import xml.etree.ElementTree as ET
import zlib
from collections import defaultdict

import orjson
from wordfreq import zipf_frequency

SCHEMA_VERSION = 2

SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value NOT NULL) WITHOUT ROWID;

-- One row per headword (case-sensitive: "bank" and "Bank" are separate rows).
-- body: zlib-compressed JSON (deflate with the preset dictionary in meta 'zdict') with everything shown on the word's page (see entry_body()).
CREATE TABLE entry (
    id INTEGER PRIMARY KEY,
    word TEXT NOT NULL,
    norm TEXT NOT NULL,      -- lower case, accents removed: what search matches against
    freq REAL NOT NULL,      -- Zipf frequency (0-8); higher = more common, used for ranking
    ipa TEXT,                -- first pronunciation, for lists ("/pɔɪzd/")
    pos TEXT,                -- first part of speech, for lists ("adj")
    rhyme TEXT,              -- ending sound, e.g. "-æŋk"; words sharing it rhyme
    anagram TEXT,            -- letters sorted; words sharing it are anagrams
    body BLOB NOT NULL
);
CREATE INDEX entry_norm ON entry (norm, freq DESC);
CREATE INDEX entry_rhyme ON entry (rhyme, freq DESC) WHERE rhyme IS NOT NULL;
CREATE INDEX entry_anagram ON entry (anagram) WHERE anagram IS NOT NULL;

-- Other spellings that lead to an entry: inflections ("ran" -> run) and alternative or
-- obsolete spellings ("dictionnary" -> dictionary).
CREATE TABLE form (
    norm TEXT NOT NULL,
    entry_id INTEGER NOT NULL,
    form TEXT NOT NULL,
    kind TEXT NOT NULL,       -- e.g. "past of", "plural of", "alternative spelling of"
    PRIMARY KEY (norm, entry_id, form)
) WITHOUT ROWID;
"""

# Sense tags that push a meaning below the everyday ones.
SINK_TAGS = {
    "obsolete", "archaic", "rare", "dated", "nonstandard", "misspelling", "proscribed",
    "historical", "uncommon", "dialectal", "obsolete-in-sense",
}
# Tags shown as labels when Wiktionary gives no written label.
LABEL_TAGS = {
    "informal", "slang", "archaic", "obsolete", "dated", "rare", "US", "UK", "British",
    "Australia", "Canada", "Ireland", "Scotland", "India", "New-Zealand", "South-Africa",
    "colloquial", "vulgar", "derogatory", "offensive", "humorous", "figuratively", "literary",
    "poetic", "formal", "dialectal", "nonstandard", "euphemistic", "Internet", "historical",
    "idiomatic", "slur", "neologism", "transitive", "intransitive", "uncountable", "countable",
}
POS_NAMES = {
    "adj": "adjective", "adv": "adverb", "intj": "interjection", "prep": "preposition",
    "conj": "conjunction", "pron": "pronoun", "det": "determiner", "num": "numeral",
    "name": "proper noun", "prep_phrase": "prepositional phrase", "adv_phrase": "adverbial phrase",
    "postp": "postposition", "punct": "punctuation",
}
WN_POS = {"n": "noun", "v": "verb", "a": "adj", "s": "adj", "r": "adv"}

# A proper name whose every meaning is one of these, with no etymology, is "empty".
EMPTY_NAME = re.compile(
    r"^(?:an? )?(?:(?:english|scottish|irish|welsh|cornish|german|french|italian|spanish|"
    r"dutch|jewish|norwegian|swedish|danish|polish|habitational|occupational|patronymic|"
    r"topographic|common|rare|chinese|korean|vietnamese|japanese|indian|hindu|muslim|"
    r"portuguese|hungarian|czech|russian|ukrainian|greek|finnish|[a-z-]+)\s+)*"
    r"(?:surname|placename|place name|census-designated place|unincorporated community|"
    r"(?:small )?(?:town|village|hamlet|city|suburb|neighbourhood|neighborhood|locality|"
    r"township|parish|civil parish|community|municipality|county|settlement|ghost town|"
    r"borough|district|area|place|river|lake|mountain|island|commune|rural municipality)"
    r"\b)",
    re.IGNORECASE,
)
NAME_FREQ_KEEP = 3.0  # Zipf; common enough to turn up in books and articles

VOWELS = set("aeiouyæɑɒɐəɘɚɛɜɝɞɤɨɪɯɵɶʉʊʌʏøœɔ")
MAX_EXAMPLES_PER_SENSE = 2
MAX_EXAMPLE_CHARS = 280
MAX_MOBY = 60


def norm(s):
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.replace("’", "'").strip()


def freq(word):
    return zipf_frequency(word, "en")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------- Wiktionary


def clean_etymology(text):
    if not text:
        return None
    lines = text.split("\n")
    if lines and lines[0].strip() == "Etymology tree":
        i = 1
        while i < len(lines) and (lines[i].strip() == "▲" or len(lines[i].split()) <= 4):
            i += 1
        lines = lines[i:]
    text = "\n".join(l for l in lines if l.strip()).strip()
    return text or None


def pick_ipa(sounds):
    out = []
    for s in sounds or []:
        ipa = s.get("ipa")
        if ipa and ipa.startswith("/") and ipa not in [o["ipa"] for o in out]:
            item = {"ipa": ipa}
            if s.get("tags"):
                item["tags"] = s["tags"]
            out.append(item)
    return out[:4]


def rhyme_of(sounds):
    for s in sounds or []:
        r = s.get("rhymes")
        if r:
            return "-" + r.lstrip("-")
    for s in sounds or []:
        ipa = s.get("ipa")
        if ipa and ipa.startswith("/"):
            return rhyme_from_ipa(ipa)
    return None


def rhyme_from_ipa(ipa):
    p = ipa.strip("/").split("/")[0].split(",")[0].strip()
    p = re.sub(r"[ˌ.()‿͡]", "", p)
    if "ˈ" in p:
        p = p[p.rindex("ˈ") + 1:]
    for i, c in enumerate(p):
        if c in VOWELS:
            r = p[i:]
            return "-" + r if r else None
    return None


def label_of(sense):
    labels = []
    raw = (sense.get("raw_glosses") or [None])[-1]
    gloss = (sense.get("glosses") or [""])[-1]
    if raw and raw.startswith("(") and not gloss.startswith("("):
        m = re.match(r"^\(([^)]*)\)", raw)
        if m:
            labels = [x.strip() for x in m.group(1).split(",") if x.strip()]
    if not labels:
        labels = [t.replace("-", " ") for t in sense.get("tags", []) if t in LABEL_TAGS]
        labels += [q for q in [sense.get("qualifier")] if q]
    return labels


YEAR = re.compile(r"^\s*(?:c\.\s*)?(\d{3,4})")


def trim_example(ex):
    text = (ex.get("text") or "").strip()
    if not text or "ſ" in text:
        return None
    offs = ex.get("bold_text_offsets") or []
    if len(text) > MAX_EXAMPLE_CHARS:
        # Keep the sentence that holds the word (the bold part).
        start = offs[0][0] if offs else 0
        bounds = [0] + [m.end() for m in re.finditer(r"[.!?][”\"’']?\s+", text)] + [len(text)]
        for a, b in zip(bounds, bounds[1:]):
            if a <= start < b:
                text = text[a:b].strip()
                break
        if len(text) > MAX_EXAMPLE_CHARS:
            return None
    text = text.strip("[…] ").strip()
    if len(text) < 12:
        return None
    out = {"t": text}
    ref = ex.get("ref")
    if ref:
        ref = ref.rstrip(":").strip()
        out["r"] = ref if len(ref) <= 200 else ref[:197].rstrip() + "…"
        m = YEAR.match(ref)
        out["_year"] = int(m.group(1)) if m else 0
    else:
        out["_year"] = 9999  # made-up usage examples are modern
    return out


def link_spans(gloss, links):
    """Words in a definition that lead to other English entries: [[start, end, target]].
    Resolved against the final word list later (see build)."""
    out, used = [], set()
    for link in links or []:
        if len(link) < 2:
            continue
        text, target = link[0], link[1]
        if "#" in target:
            target, lang = target.split("#", 1)
            if lang and not lang.startswith("English"):
                continue
        if ":" in target or not text or not target:
            continue
        for m in re.finditer(r"(?<![\w-])" + re.escape(text) + r"(?![\w-])", gloss):
            if m.start() not in used:
                used.add(m.start())
                out.append([m.start(), m.end(), target])
                break
    return out


def pick_examples(sense):
    exs = [e for e in (trim_example(x) for x in sense.get("examples", [])) if e]
    # Prefer modern writing: usage examples first, then newer quotations.
    exs.sort(key=lambda e: -e["_year"])
    for e in exs:
        e.pop("_year")
    return exs[:MAX_EXAMPLES_PER_SENSE]


def words_of(items):
    seen, out = set(), []
    for it in items or []:
        w = it.get("word")
        if w and w not in seen:
            seen.add(w)
            out.append(w)
    return out


FORM_KIND = {  # checked in this order
    "past": "past of", "third-person": "present of", "gerund": "present participle of",
    "comparative": "comparative of", "superlative": "superlative of",
    "plural": "plural of", "participle": "participle of",
}


def form_kind(tags):
    if "alt-of" in tags or "alternative" in tags:
        return "alternative spelling of"
    if "misspelling" in tags:
        return "misspelling of"
    if "abbreviation" in tags or "initialism" in tags or "acronym" in tags:
        return "short for"
    if "participle" in tags and "past" in tags:
        return "past participle of"
    if "participle" in tags and "present" in tags:
        return "present participle of"
    for t, k in FORM_KIND.items():
        if t in tags:
            return k
    return "form of"


def slim_entry(d):
    """Keeps what Alphabet shows from one Wiktionary part-of-speech entry."""
    senses = []
    links = []  # (target word, kind) for form-of / alt-of senses
    for s in d.get("senses", []):
        tags = set(s.get("tags", []))
        if "no-gloss" in tags or not s.get("glosses"):
            continue
        target = (s.get("form_of") or s.get("alt_of") or [None])[0]
        if target and target.get("word"):
            links.append((target["word"], form_kind(tags)))
            # An abbreviation's gloss is a real meaning, keep it; pure inflections are links only.
            if not ({"abbreviation", "initialism", "acronym"} & tags):
                continue
        glosses = s["glosses"]
        sense = {"g": glosses[-1]}
        if len(glosses) > 1:
            sense["parent"] = glosses[-2]
        lab = label_of(s)
        if lab:
            sense["l"] = lab
        ex = pick_examples(s)
        if ex:
            sense["x"] = ex
        syn = words_of(s.get("synonyms"))
        if syn:
            sense["syn"] = syn[:20]
        spans = link_spans(glosses[-1], s.get("links"))
        if spans:
            sense["k"] = spans
        ant = words_of(s.get("antonyms"))
        if ant:
            sense["ant"] = ant[:12]
        sense["_sink"] = bool(tags & SINK_TAGS)
        senses.append(sense)

    # Common meanings first; obsolete, archaic and rare ones after (order kept otherwise).
    senses.sort(key=lambda s: s["_sink"])
    sunk = all(s.pop("_sink") for s in senses) if senses else True
    for s in senses:
        s.pop("_sink", None)

    alts, forms, seen = [], [], set()
    for f in d.get("forms", []):
        tags = f.get("tags", [])
        form = f.get("form")
        # Skip full conjugation tables: they repeat forms with every person and number.
        if (not form or form == d["word"] or form in seen or f.get("source")
                or "table-tags" in tags or "inflection-template" in tags):
            continue
        seen.add(form)
        if "alternative" in tags:
            alts.append(form)
        elif not ({"romanization", "obsolete", "archaic"} & set(tags)):
            forms.append({"f": form, "t": [t for t in tags if t not in ("canonical",)]})

    entry = {"pos": d.get("pos")}
    ipa = pick_ipa(d.get("sounds"))
    if ipa:
        entry["ipa"] = ipa
    ety = clean_etymology(d.get("etymology_text"))
    if ety:
        entry["ety"] = ety
    if forms:
        entry["forms"] = forms[:12]
    if alts:
        entry["alt"] = alts[:10]
    for key, src in (("syn", "synonyms"), ("ant", "antonyms"), ("rel", "related"),
                     ("der", "derived"), ("hyper", "hypernyms"), ("hypo", "hyponyms")):
        ws = words_of(d.get(src))
        if ws:
            entry[key] = ws[:40]
    entry["senses"] = senses
    return entry, links, sunk, rhyme_of(d.get("sounds"))


def is_empty_name(d, entry):
    if d.get("pos") != "name" or entry.get("ety"):
        return False
    glosses = [s["g"] for s in entry["senses"]]
    return bool(glosses) and all(EMPTY_NAME.match(g) for g in glosses)


def read_wiktionary(path, stats):
    words = defaultdict(list)     # word -> [entry]
    rhymes = {}
    links = defaultdict(set)      # source word -> {(target, kind)}
    with open(path, "rb") as f:
        for n, line in enumerate(f, 1):
            if n % 200000 == 0:
                log(f"  wiktionary {n:,} lines")
            d = orjson.loads(line)
            if d.get("lang_code") != "en":
                continue
            stats["wikt_entries"] += 1
            word = d["word"]
            entry, lk, sunk, rhyme = slim_entry(d)
            for target, kind in lk:
                if target != word:
                    links[word].add((target, kind))
            if not entry["senses"]:
                stats["link_only_entries"] += 1
                continue
            if is_empty_name(d, entry) and freq(word) < NAME_FREQ_KEEP:
                stats["empty_names_dropped"] += 1
                continue
            entry["_sunk"] = sunk
            words[word].append(entry)
            if rhyme and word not in rhymes and " " not in word:
                rhymes[word] = rhyme
    return words, rhymes, links


# ---------------------------------------------------------------- WordNet


def read_wordnet(path):
    """Returns {(lemma, pos): [group]} with meaning-grouped synonyms, broader and narrower
    words, and antonyms (stretched through similar-adjective groups)."""
    entry_lemma, sense_info, synsets = {}, {}, {}
    lemma_senses = defaultdict(list)  # (lemma, pos) -> [sense id] in WordNet order
    with gzip.open(path) as f:
        for _, el in ET.iterparse(f, events=("end",)):
            tag = el.tag
            if tag == "LexicalEntry":
                lem = el.find("Lemma")
                form, pos = lem.get("writtenForm"), WN_POS[lem.get("partOfSpeech")]
                entry_lemma[el.get("id")] = form
                for s in el.findall("Sense"):
                    ants = [r.get("target") for r in s.findall("SenseRelation") if r.get("relType") == "antonym"]
                    sense_info[s.get("id")] = (form, s.get("synset"), ants)
                    lemma_senses[(form, pos)].append(s.get("id"))
                el.clear()
            elif tag == "Synset":
                rels = defaultdict(list)
                for r in el.findall("SynsetRelation"):
                    rels[r.get("relType")].append(r.get("target"))
                d = el.find("Definition")
                synsets[el.get("id")] = {
                    "members": el.get("members", "").split(),
                    "pos": el.get("partOfSpeech"),
                    "def": d.text if d is not None else "",
                    "rels": rels,
                }
                el.clear()

    def lemmas(sid):
        return [entry_lemma[m] for m in synsets[sid]["members"] if m in entry_lemma]

    # Antonyms of each synset, from its senses.
    synset_ants = defaultdict(set)
    for sid_sense, (form, sid, ants) in sense_info.items():
        for a in ants:
            if a in sense_info:
                synset_ants[sid].add(sense_info[a][1])

    out = defaultdict(list)
    for (form, pos), sense_ids in lemma_senses.items():
        for sense_id in sense_ids:
            _, sid, ants = sense_info[sense_id]
            ss = synsets[sid]
            syn = [l for l in lemmas(sid) if l != form]
            ant = [sense_info[a][0] for a in ants if a in sense_info]
            similar = []
            if ss["pos"] == "s":
                # Satellite adjective: borrow the head's antonyms (hot <-> cold gives
                # scorching <-> cold) and list the head as a similar word.
                for head in ss["rels"].get("similar", []):
                    similar += lemmas(head)
                    for anti in synset_ants.get(head, ()):
                        ant += lemmas(anti)
            broader = [l for h in ss["rels"].get("hypernym", []) for l in lemmas(h)[:3]]
            narrower = [l for h in ss["rels"].get("hyponym", []) for l in lemmas(h)[:2]]
            group = {"d": ss["def"]}
            for key, vals, cap in (("syn", syn + similar, 20), ("ant", ant, 10),
                                   ("broader", broader, 6), ("narrower", narrower, 16)):
                vals = [v for v in dict.fromkeys(vals) if v != form][:cap]
                if vals:
                    group[key] = vals
            if len(group) > 1:
                out[(form, pos)].append(group)
    return out


def read_moby(path):
    out = {}
    with open(path, encoding="latin-1") as f:
        for line in f:
            parts = line.rstrip("\r\n").split(",")
            if len(parts) > 1:
                out[parts[0]] = parts[1:]
    return out


# ---------------------------------------------------------------- Build


def train_zdict(words, size=32768, seed=7):
    """A preset dictionary for zlib: a sample of typical entry text that every entry can
    refer back to. Shrinks entries by about 30%. Android reads it with
    Inflater.setDictionary(); it is stored in meta under 'zdict'."""
    import random
    rnd = random.Random(seed)
    sample = rnd.sample(sorted(words), min(4000, len(words)))
    blob = b"".join(orjson.dumps({"w": w, "p": words[w]}) for w in sample)
    return blob[-size:]


def compress(data, zdict):
    c = zlib.compressobj(9, zlib.DEFLATED, 15, 9, zlib.Z_DEFAULT_STRATEGY, zdict)
    return c.compress(data) + c.flush()


def anagram_key(word):
    if not re.fullmatch(r"[A-Za-z]+", word) or len(word) < 3:
        return None
    return "".join(sorted(word.lower()))


def build(raw_dir, out_path):
    stats = defaultdict(int)
    log("Reading Wiktionary")
    words, rhymes, links = read_wiktionary(os.path.join(raw_dir, "wiktextract-en.jsonl"), stats)
    log("Reading WordNet")
    wordnet = read_wordnet(os.path.join(raw_dir, "oewn-2025.xml.gz"))
    log("Reading Moby")
    moby = read_moby(os.path.join(raw_dir, "moby", "files", "mthesaur.txt"))

    if os.path.exists(out_path):
        os.remove(out_path)
    db = sqlite3.connect(out_path)
    db.executescript("PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; PRAGMA page_size=4096;")
    db.executescript(SCHEMA)

    log("Training the compression dictionary")
    zdict = train_zdict(words)
    log("Writing entries")
    known = set(words)
    ids = {}
    order = sorted(words, key=lambda w: (norm(w), w))
    raw_bytes = 0
    for i, word in enumerate(order, 1):
        entries = words[word]
        # Parts of speech with only obsolete or rare meanings go last.
        entries.sort(key=lambda e: e.pop("_sunk"))
        body = {"w": word, "p": entries}

        wn_groups = []
        for pos in dict.fromkeys(e["pos"] for e in entries):
            for g in wordnet.get((word, pos), []):
                g["pos"] = pos
                wn_groups.append(g)
        if wn_groups:
            body["th"] = wn_groups
            stats["with_wordnet"] += 1

        listed = {word}
        for e in entries:
            listed.update(e.get("syn", []))
            for s in e["senses"]:
                listed.update(s.get("syn", []))
        for g in wn_groups:
            listed.update(g.get("syn", []))
        more = [m for m in moby.get(word, []) if m not in listed and m in known]
        if more:
            more.sort(key=lambda m: -freq(m))
            body["more"] = more[:MAX_MOBY]
            stats["with_moby"] += 1

        # Keep only links to words in the dictionary (and not to the word itself).
        for e in entries:
            for s in e["senses"]:
                if "k" in s:
                    s["k"] = [k for k in s["k"] if k[2] in known and k[2] != word]
                    if not s["k"]:
                        del s["k"]
                    else:
                        stats["senses_with_links"] += 1
        if any(s.get("x") for e in entries for s in e["senses"]):
            stats["with_example"] += 1
        if any(e.get("ety") for e in entries):
            stats["with_etymology"] += 1
        if any(e.get("ipa") for e in entries):
            stats["with_ipa"] += 1
        if any(s.get("syn") for e in entries for s in e["senses"]) or any(e.get("syn") for e in entries) or wn_groups:
            stats["with_synonyms"] += 1

        data = orjson.dumps(body)
        raw_bytes += len(data)
        db.execute(
            "INSERT INTO entry (id, word, norm, freq, ipa, pos, rhyme, anagram, body) VALUES (?,?,?,?,?,?,?,?,?)",
            (i, word, norm(word), round(freq(word), 2),
             next((e["ipa"][0]["ipa"] for e in entries if e.get("ipa")), None), entries[0]["pos"],
             rhymes.get(word), anagram_key(word),
             compress(data, zdict)),
        )
        ids[word] = i
    stats["entries"] = len(order)

    log("Writing forms")
    rows = {}
    # Form-of senses first ("ran": simple past of run): their labels are the most precise.
    for src, targets in links.items():
        for target, kind in sorted(targets):
            if target in ids:
                rows.setdefault((norm(src), ids[target], src), kind)
    # Inflected forms listed under each entry ("dictionaries" -> dictionary).
    for word, entries in words.items():
        for e in entries:
            for f in e.get("forms", []):
                rows.setdefault((norm(f["f"]), ids[word], f["f"]), form_kind(set(f["t"])))
            for a in e.get("alt", []):
                rows.setdefault((norm(a), ids[word], a), "alternative spelling of")
    rows = [(*k, v) for k, v in rows.items() if k[0]]
    db.executemany("INSERT OR IGNORE INTO form VALUES (?,?,?,?)", rows)
    stats["forms"] = len(rows)

    meta = {
        "schema_version": str(SCHEMA_VERSION),
        "built": time.strftime("%Y-%m-%d"),
        "sources": "Wiktionary (via Wiktextract, kaikki.org, 2026-10-30 extraction); "
                   "Open English WordNet 2025; Moby Thesaurus II; wordfreq",
        "licence": "CC BY-SA 4.0. Adapted from Wiktionary (CC BY-SA), Open English WordNet "
                   "(CC BY 4.0), Moby Thesaurus (public domain) and wordfreq data (CC BY-SA 4.0).",
    }
    db.executemany("INSERT INTO meta VALUES (?,?)", meta.items())
    db.execute("INSERT INTO meta VALUES ('zdict', ?)", (zdict,))
    db.commit()
    log("Vacuuming")
    db.execute("VACUUM")
    db.close()
    stats["json_bytes_uncompressed"] = raw_bytes
    stats["db_bytes"] = os.path.getsize(out_path)
    for k in sorted(stats):
        print(f"  {k}: {stats[k]:,}")
    return stats


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
