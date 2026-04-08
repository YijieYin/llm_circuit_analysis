"""
synonym_table.py — Drosophila MB cell type synonym mappings.

Sources:
- Figs 6–8 + supplement of eLife 62576 (Takemura et al. / hemibrain paper)
  giving connectome names (PPL1xx, PAMxx, MBONxx) and their compartment-based aliases.
- KC names from connectome flat list (with p = prime).

Each entry: alias (as it might appear in literature) → connectome canonical name.
Values are lists so multiple aliases map to the same canonical.
Greek letters: we include both unicode (α β γ etc.) and ASCII (a b g etc.) variants.
"""


# ── helpers ──────────────────────────────────────────────────────────────────
def _greek(s):
    """Return both greek-unicode and ascii-letter variants of a string."""
    greek_to_ascii = {
        "α": "a",
        "β": "b",
        "γ": "g",
        "δ": "d",
        "ε": "e",
        "α′": "a'",
        "α'": "a'",
        "β′": "b'",
        "β'": "b'",
        "′": "'",
    }
    ascii_s = s
    for g, a in greek_to_ascii.items():
        ascii_s = ascii_s.replace(g, a)
    variants = [s]
    if ascii_s != s:
        variants.append(ascii_s)
    return variants


def _expand(canonical, aliases):
    """For each alias, add greek/ascii variants. Returns list of (alias, canonical) pairs."""
    pairs = []
    for a in aliases:
        for v in _greek(a):
            pairs.append((v, canonical))
    return pairs


# ── build synonym dict ────────────────────────────────────────────────────────
# Format: alias_string -> connectome_canonical_name
# We build it as a flat dict (alias → canonical).

_entries = []

# ── PPL1 DANs ────────────────────────────────────────────────────────────────
_entries += _expand(
    "PPL101",
    [
        "PPL101",
        "PPL1-γ1pedc",
        "PPL1-γ1ped",
        "PPL-γ1pedc",
        "PPL-γ1ped",
        "PPL1 γ1pedc",
        "PPL1-g1pedc",
        "PPL1-g1ped",
        "DAN-PPL1-γ1pedc",
        "PPL1-γ1pedc dopaminergic neuron",
        "MB-MP1",
        "MB-MP1 dopamine neuron",
        "PPL1 dopamine neuron γ1pedc",
        "PPL1-γ1ped>α/β",
    ],
)
_entries += _expand(
    "PPL102",
    [
        "PPL102",
        "PPL1-γ1",
        "PPL-γ1",
        "PPL1 γ1",
        "PPL1-g1",
        "DAN-PPL1-γ1",
    ],
)
_entries += _expand(
    "PPL103",
    [
        "PPL103",
        "PPL1-γ2α'1",
        "PPL1-γ2α′1",
        "PPL-γ2α'1",
        "PPL1-g2a1",
        "PPL1-γ2a'1",
        "DAN-PPL1-γ2α'1",
        "MB-MV1",
        "MB-MV1 dopamine neuron",
        "PPL1 γ2α'1",
    ],
)
_entries += _expand(
    "PPL104",
    [
        "PPL104",
        "PPL1-α'3",
        "PPL1-α′3",
        "PPL-α'3",
        "PPL1-a'3",
        "DAN-PPL1-α'3",
        "PPL1 α'3",
    ],
)
_entries += _expand(
    "PPL105",
    [
        "PPL105",
        "PPL1-α'2α2",
        "PPL1-α′2α2",
        "PPL-α'2α2",
        "PPL1-a'2a2",
        "PPL1-α2α'2",
        "DAN-PPL1-α'2α2",
        "PPL1 α'2α2",
    ],
)
_entries += _expand(
    "PPL106",
    [
        "PPL106",
        "PPL1-α3",
        "PPL-α3",
        "PPL1-a3",
        "DAN-PPL1-α3",
        "PPL1 α3",
        "DAN PPL1-α3",
    ],
)

# ── PAM DANs ─────────────────────────────────────────────────────────────────
_entries += _expand(
    "PAM01",
    [
        "PAM01",
        "PAM-γ5",
        "PAM-g5",
        "PAM γ5",
        "PAM-γ5 dopamine",
        "aSP13",
        "DAN-aSP13",
        "PAM-γ5/aSP13",
    ],
)
_entries += _expand(
    "PAM02",
    [
        "PAM02",
        "PAM-β'2α",
        "PAM-β′2α",
        "PAM-b'2a",
        "PAM-β'2a",
        "PAM β'2α",
    ],
)
_entries += _expand(
    "PAM03",
    [
        "PAM03",
        "PAM-β2β'2α",
        "PAM-β2β′2α",
        "PAM-b2b'2a",
        "PAM-β2β'2a",
    ],
)
_entries += _expand(
    "PAM04",
    [
        "PAM04",
        "PAM-β2",
        "PAM-b2",
        "PAM β2",
    ],
)
_entries += _expand(
    "PAM05",
    [
        "PAM05",
        "PAM-β'2p",
        "PAM-β′2p",
        "PAM-b'2p",
        "PAM-β'2mp",
        "PAM-β′2mp",
    ],
)
_entries += _expand(
    "PAM06",
    [
        "PAM06",
        "PAM-β'2m",
        "PAM-β′2m",
        "PAM-b'2m",
    ],
)
_entries += _expand(
    "PAM07",
    [
        "PAM07",
        "PAM-γ4<γ1γ2",
        "PAM-γ4>γ1γ2",
        "PAM-g4>g1g2",
        "PAM-γ4<γ1γ2",
        "PAM γ4>γ1γ2",
    ],
)
_entries += _expand(
    "PAM08",
    [
        "PAM08",
        "PAM-γ4",
        "PAM-g4",
        "PAM γ4",
        "PAM08_a(y4)",
        "PAM08_b(y4)", 
        "PAM08_c(y4)", 
        "PAM08_e(y4)"
    ],
)
_entries += _expand(
    "PAM09",
    [
        "PAM09",
        "PAM-β1ped",
        "PAM-b1ped",
        "PAM-βped",
        "PAM-b1p",
    ],
)
_entries += _expand(
    "PAM10",
    [
        "PAM10",
        "PAM-β1",
        "PAM-b1",
        "PAM β1",
    ],
)
_entries += _expand(
    "PAM11",
    [
        "PAM11",
        "PAM-α1",
        "PAM-a1",
        "PAM α1",
        "PAM-α1 dopamine",
    ],
)
_entries += _expand(
    "PAM12",
    [
        "PAM12",
        "PAM-γ3",
        "PAM-g3",
        "PAM γ3",
    ],
)
_entries += _expand(
    "PAM13",
    [
        "PAM13",
        "PAM-β'1ap",
        "PAM-β′1ap",
        "PAM-b'1ap",
    ],
)
_entries += _expand(
    "PAM14",
    [
        "PAM14",
        "PAM-β'1m",
        "PAM-β′1m",
        "PAM-b'1m",
    ],
)
_entries += _expand(
    "PAM15",
    [
        "PAM15",
        "PAM-γ5β'2a",
        "PAM-γ5β′2a",
        "PAM-g5b'2a",
    ],
)

# ── MBONs ─────────────────────────────────────────────────────────────────────
_entries += _expand(
    "MBON01",
    [
        "MBON01",
        "MBON-γ5β'2a",
        "MBON-γ5β′2a",
        "MBON-g5b'2a",
        "M6",
        "MB-M6",
        "M6 neuron",
    ],
)
_entries += _expand(
    "MBON02",
    [
        "MBON02",
        "MBON-β2β'2a",
        "MBON-β2β′2a",
        "MBON-b2b'2a",
    ],
)
_entries += _expand(
    "MBON03",
    [
        "MBON03",
        "MBON-β'2mp",
        "MBON-β′2mp",
        "MBON-b'2mp",
    ],
)
_entries += _expand(
    "MBON04",
    [
        "MBON04",
        "MBON-β'2mp_bilateral",
        "MBON-β′2mp_bilateral",
        "MBON-b'2mp_bilateral",
    ],
)
_entries += _expand(
    "MBON05",
    [
        "MBON05",
        "MBON-γ4>γ1γ2",
        "MBON-g4>g1g2",
        "MBON-γ4<γ1γ2",
    ],
)
_entries += _expand(
    "MBON06",
    [
        "MBON06",
        "MBON-β1>α",
        "MBON-β>α1",
        "MBON-b1>a",
        "MBON-β>a1",
        "MBON-β1→α",
    ],
)
_entries += _expand(
    "MBON07",
    [
        "MBON07",
        "MBON-α1",
        "MBON-a1",
        "MBON α1",
        "mushroom body output neuron α1",
    ],
)
_entries += _expand(
    "MBON08",
    [
        "MBON08",
        "MBON-γ3",
        "MBON-g3",
    ],
)
_entries += _expand(
    "MBON09",
    [
        "MBON09",
        "MBON-γ3β'1",
        "MBON-γ3β′1",
        "MBON-g3b'1",
    ],
)
_entries += _expand(
    "MBON10",
    [
        "MBON10",
        "MBON-β'1",
        "MBON-β′1",
        "MBON-b'1",
    ],
)
_entries += _expand(
    "MBON11",
    [
        "MBON11",
        "MBON-γ1pedc>α/β",
        "MBON-γ1ped>α/β",
        "MBON-g1pedc>a/b",
        "MBON-γ1pedc",
        "MVP2",
        "MB-MVP2",
        "MBON-112c" # based on https://elifesciences.org/articles/04580#:~:text=For%20aversive%20visual%20memory%20(Figure,9%E2%80%94figure%20supplement%201).
    ],
)
_entries += _expand(
    "MBON12",
    [
        "MBON12",
        "MBON-γ2α'1",
        "MBON-γ2α′1",
        "MBON-g2a'1",
        "MBON-γ2a'1",
    ],
)
_entries += _expand(
    "MBON13",
    [
        "MBON13",
        "MBON-α'2",
        "MBON-α′2",
        "MBON-a'2",
    ],
)
_entries += _expand(
    "MBON14",
    [
        "MBON14",
        "MBON-α3",
        "MBON-a3",
    ],
)
_entries += _expand(
    "MBON15",
    [
        "MBON15",
        "MBON-α'1",
        "MBON-α′1",
        "MBON-a'1",
        "MBON15-like",
        "MBON-α'1α'2",
    ],
)
_entries += _expand(
    "MBON16",
    [
        "MBON16",
        "MBON-α'3ap",
        "MBON-α′3ap",
        "MBON-a'3ap",
    ],
)
_entries += _expand(
    "MBON17",
    [
        "MBON17",
        "MBON-α'3m",
        "MBON-α′3m",
        "MBON-a'3m",
        "MBON-α'2α'3",
        "MBON17-like",
    ],
)
_entries += _expand(
    "MBON18",
    [
        "MBON18",
        "MBON-α2sc",
        "MBON-a2sc",
    ],
)
_entries += _expand(
    "MBON19",
    [
        "MBON19",
        "MBON-α2p3p",
        "MBON-a2p3p",
    ],
)
_entries += _expand(
    "MBON20",
    [
        "MBON20",
        "MBON-γ1γ2",
        "MBON-g1g2",
    ],
)
_entries += _expand(
    "MBON21",
    [
        "MBON21",
        "MBON-γ4γ5",
        "MBON-g4g5",
    ],
)
_entries += _expand(
    "MBON22",
    [
        "MBON22",
        "MBON-calyx",
        "MBON calyx",
    ],
)
_entries += _expand(
    "MBON23",
    [
        "MBON23",
        "MBON-α2sp",
        "MBON-a2sp",
    ],
)
_entries += _expand(
    "MBON24",
    [
        "MBON24",
        "MBON-β2γ5",
        "MBON-b2g5",
    ],
)
_entries += _expand(
    "MBON25",
    [
        "MBON25",
        "MBON-γ1γ2",  # atypical, same compartment as MBON20 — keep both
    ],
)
_entries += _expand(
    "MBON26",
    [
        "MBON26",
        "MBON-β'2d",
        "MBON-β′2d",
        "MBON-b'2d",
    ],
)
_entries += _expand(
    "MBON27",
    [
        "MBON27",
        "MBON-γ5d",
        "MBON-g5d",
    ],
)
_entries += _expand(
    "MBON28",
    [
        "MBON28",
        "MBON-α'3a",
        "MBON-α′3a",
        "MBON-a'3a",
    ],
)
_entries += _expand(
    "MBON29",
    [
        "MBON29",
        "MBON-γ4γ5",  # atypical; same compartment as MBON21
    ],
)
_entries += _expand(
    "MBON30",
    [
        "MBON30",
        "MBON-γ1γ2γ3",
        "MBON-g1g2g3",
    ],
)
_entries += _expand(
    "MBON31",
    [
        "MBON31",
        "MBON-α'1a",
        "MBON-α′1a",
        "MBON-a'1a",
    ],
)
_entries += _expand(
    "MBON32",
    [
        "MBON32",
        "MBON-γ2",  # atypical
    ],
)
_entries += _expand(
    "MBON33",
    [
        "MBON33",
        "MBON-γ2γ3",
        "MBON-g2g3",
    ],
)
_entries += _expand(
    "MBON34",
    [
        "MBON34",
        "MBON-γ2",  # atypical; note MBON32 and 34 both γ2 — keep separate
    ],
)
_entries += _expand(
    "MBON35",
    [
        "MBON35",
        "MBON-γ2",  # atypical
    ],
)

# ── Kenyon Cells ──────────────────────────────────────────────────────────────
# Connectome names: KCab, KCg-m, KCapbp-ap1, KCapbp-m, KCab-p,
#                   KCapbp-ap2, KCg-d, KCa'b'-ap1, KCg-s3, KCg-s1, KCg-s2
# (p = prime in the flat list encoding)

_entries += _expand(
    "KCab",
    [
        "KCab",
        "KC α/β",
        "KC αβ",
        "KCα/β",
        "KCαβ",
        "α/β Kenyon cell",
        "αβ Kenyon cell",
        "MB αβ neuron",
        "MB α/β neuron",
        "Kenyon cells α/β",
        "KCa/b",
    ],
)
_entries += _expand(
    "KCab-p",
    [
        "KCab-p",
        "KC α/β posterior",
        "KCαβp",
        "KCα/βp",
        "αβp Kenyon cell",
        "MB αβp neuron",
        "αβ posterior Kenyon cell",
    ],
)
_entries += _expand(
    "KCa'b'-ap1",
    [
        "KCa'b'-ap1",
        "KCapbp-ap1",
        "KC α'β' ap1",
        "KCα'β'ap1",
        "α'β' ap1 Kenyon cell",
        "MB α'β' ap1 neuron",
        "KCα'/β'ap1",
        "KC α'/β' ap1",
    ],
)
_entries += _expand(
    "KCapbp-ap2",
    [
        "KCapbp-ap2",
        "KC α'β' ap2",
        "KCα'β'ap2",
        "α'β' ap2 Kenyon cell",
        "MB α'β' ap2 neuron",
    ],
)
_entries += _expand(
    "KCapbp-m",
    [
        "KCapbp-m",
        "KC α'β' m",
        "KCα'β'm",
        "KC α'/β' medial",
        "α'β'm Kenyon cell",
        "MB α'β' medial neuron",
        "KCα'/β'm",
    ],
)
_entries += _expand(
    "KCg-m",
    [
        "KCg-m",
        "KC γ main",
        "KCγm",
        "KCγ-m",
        "γm Kenyon cell",
        "MB γ main neuron",
        "KC g-m",
        "γ main KC",
    ],
)
_entries += _expand(
    "KCg-d",
    [
        "KCg-d",
        "KC γ dorsal",
        "KCγd",
        "KCγ-d",
        "γd Kenyon cell",
        "MB γ dorsal neuron",
        "KC g-d",
        "γd KC",
        "γ dorsal KC",
    ],
)
_entries += _expand(
    "KCg-s1",
    [
        "KCg-s1",
        "KC γs1",
        "KCγs1",
        "KCγ-s1",
        "γs1 Kenyon cell",
    ],
)
_entries += _expand(
    "KCg-s2",
    [
        "KCg-s2",
        "KC γs2",
        "KCγs2",
        "KCγ-s2",
        "γs2 Kenyon cell",
    ],
)
_entries += _expand(
    "KCg-s3",
    [
        "KCg-s3",
        "KC γs3",
        "KCγs3",
        "KCγ-s3",
        "γs3 Kenyon cell",
    ],
)

# ── LH neurons — literature often omits "LH" prefix ─────────────────────────
# Selected examples from the CSV (AV1a1, PD2a1/b1, PV5a1, AV6a1, AD1b2, etc.)
# Connectome names are LHAV1a1, LHPD2a1, LHPD2b1, LHPV5a1, LHAV6a1, LHAD1b2 etc.

for _prefix in [
    "AV1a1",
    "AV2a2",
    "AV4a5",
    "AV6a1",
    "AD1b2",
    "AD3b1",
    "PD2a1",
    "PD2b1",
    "PV4a1",
    "PV5a1",
    "PV5k1",
]:
    _entries.append((_prefix, f"LH{_prefix}"))
    _entries.append((f"LH{_prefix}", f"LH{_prefix}"))
    # with "(LHON)" or "(LHLN)" suffix
    _entries.append((f"{_prefix} (LHON)", f"LH{_prefix}"))
    _entries.append((f"{_prefix} (LHLN)", f"LH{_prefix}"))
    _entries.append((f"LH{_prefix} (LHON)", f"LH{_prefix}"))
    _entries.append((f"LH{_prefix} (LHLN)", f"LH{_prefix}"))

# PD2a1/b1 combined
_entries += [
    ("PD2a1/b1", "LHPD2a1"),  # map to the first; b1 is separate
    ("PD2a1/b1 LHNs", "LHPD2a1"),
    ("LHN1 (PD2a1/b1)", "LHPD2a1"),
    ("LHN2 (PV5a1)", "LHPV5a1"),
    ("LHN1", "LHPD2a1"),
    ("LHN2", "LHPV5a1"),
]

# LHAV1a1 — appeared in CSV as "AV1a1 (LHON)"
_entries += _expand(
    "LHAV1a1",
    [
        "LHAV1a1",
        "AV1a1",
        "AV1a1 (LHON)",
    ],
)

# ── DA glomerulus PNs / ORNs ─────────────────────────────────────────────────
_entries += [
    ("DA1 lPNs", "DA1_lPN"),
    ("DA1 lPN", "DA1_lPN"),
    ("DA1_lPNs", "DA1_lPN"),
    ("DA1 lvPNs", "DA1_lvPN"),
    ("DA1 lvPN", "DA1_lvPN"),
    ("DA1-PNs", "DA1_lPN"),
    ("DA1-ORNs", "DA1_ORN"),
    ("DA1 ORNs", "DA1_ORN"),
]

# ── CX ─────────────────────────────────────────────────
_entries += _expand(
    "EPG",
    [
        "E-PG",
        "E-PG neurons",
        "E-PG Neurons",
        "E-PG (Compass Neurons)",
        "E-PG neurons (PBG1–8.b-EBw.s-D/Vgall.b)",
        "Compass neurons (EPG neurons)",
        "EPG neurons (Compass neurons)",
        "E-PG neurons (ellipsoid body-protocerebral bridge-gall neurons)",
    ],
)

_entries += [
    ("R5 Ring Neurons", "ER5"),
    ("ER3d ring neurons", "ER3d"),
    ("R4d Ring Neurons", "ER4d"),
]

# ── build final dict ──────────────────────────────────────────────────────────
SYNONYMS: dict[str, str] = {}
for alias, canonical in _entries:
    # store lower-stripped version for lookup; keep original for display
    SYNONYMS[alias] = canonical
    SYNONYMS[alias.strip()] = canonical

if __name__ == "__main__":
    import json

    print(f"Total synonym entries: {len(SYNONYMS)}")
    # print a sample
    sample = list(SYNONYMS.items())[:20]
    for k, v in sample:
        print(f"  {k!r:50s} -> {v}")
