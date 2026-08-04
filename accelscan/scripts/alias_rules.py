"""Alias-generation rules shared by every registry generator.

Both generators used to carry their own bare-code policy: `build_registry.py`
gated any short code it found in a Wikipedia table, while
`build_epoch_registry.py` refused bare codes under four characters outright. Two
rules over one namespace produced a recall asymmetry between the same vendor's
parts depending on which table they came from (a bare gated `K80` from Wikipedia,
but `T4` reachable only as "Tesla T4"), which is exactly the kind of arbitrary
seam a reviewer should object to. This module holds ONE rule, applied by both:

    A bare short code gets a gated alias, unless the code appears in
    BARE_CODE_DENY or another entry already claims that exact string.

The deny list is the enumerated exception, and every entry names the
non-hardware meaning that earns it a place -- a fact about the string, not about
the vendor. A length threshold was rejected as the rule: it is simpler to state
but drops `K80`, `K20`, `K40`, `P40`, `A30`, `A40` and `L40`, all real
datacenter parts with substantial paper support, while keeping equally ambiguous
four-character codes.

The second rule here is the case rule for vendor-qualified aliases. Registry
`case: auto` makes an all-uppercase pattern case-sensitive, which is right for a
bare `V100` and wrong for `NVIDIA L4`: "Nvidia L4" is the ordinary spelling and
would never have matched. Several of the affected patterns are the vendor-
qualified-only export-control SKUs (A800, HGX H20, L4), so the bug was a
directional recall bias on a reported series.
"""

import re

GATE = 'gpu'
# Shortest string we will ever emit bare.
MIN_BARE_CODE_LEN = 2

# Codes whose bare form collides with a high-frequency non-hardware meaning in a
# scientific corpus one third medicine. The ±250-character context gate does not
# rescue these: "L2 regularization ... trained on a GPU" satisfies it in one
# sentence. Such models stay reachable through a vendor-qualified alias
# ("NVIDIA L4", "Tesla T4"), which is unambiguous.
BARE_CODE_DENY = {
    'T4': 'thyroxine, the T4 thoracic vertebra, and bacteriophage T4',
    'L2': 'L2 regularization, the L2 norm, L2 cache, and the L2 vertebra',
    'L4': 'the L4 lumbar vertebra',
    'A2': 'the A2 beta-casein variant and A2 allele naming',
    'H20': 'water, written H20 whenever the subscript is lost in extraction',
    'MI6': 'the intelligence agency; MI alone is myocardial infarction',
    'MI8': 'the intelligence agency; MI alone is myocardial infarction',
    'A10': 'the A-10 aircraft and the A10 cell line',
    'C60': 'buckminsterfullerene',
    'P4': 'progesterone, and the P4 allotrope of phosphorus',
}

ALL_DIGITS = re.compile(r'^\d+$')


def deny_reason(code: str) -> str | None:
    """The stated non-hardware meaning that blocks a bare alias, or None."""
    if ALL_DIGITS.match(code):
        # An all-digit code ('7210' for a Xeon Phi, '910' for an Ascend) is
        # indistinguishable from a quantity, and no context vocabulary fixes that:
        # "of the 910 images ... on a GPU" satisfies the gate. Such parts stay
        # reachable through their product line ("Xeon Phi 7210", "Ascend 910").
        return 'an all-digit code is a number in prose'
    for denied, reason in BARE_CODE_DENY.items():
        if code.upper() == denied:
            return reason
    return None


def claimed_elsewhere(code: str, entries: dict) -> bool:
    """True if some entry already carries this exact bare pattern.

    Intel's Arc Pro A40 and NVIDIA's A40 are different GPUs with the same short
    code, and the NVIDIA part carries ~2,600 papers. Two entries claiming one
    span makes `normalize.canonicalize_one`'s tie-break arbitrary, so the second
    claimant gets no bare alias and stays reachable as 'Arc Pro A40'.
    """
    target = re.escape(code)
    for e in entries.values():
        for a in e.get('aliases', []):
            if (a if isinstance(a, str) else a['pattern']) == target:
                return True
    return False


def bare_code_aliases(code: str, entries: dict, gate: str = GATE) -> list:
    """The one bare-code rule. Returns [] or a single gated alias."""
    if (len(code) < MIN_BARE_CODE_LEN or deny_reason(code)
            or claimed_elsewhere(code, entries)):
        return []
    return [{'pattern': re.escape(code), 'gate': gate}]


# Vendor spellings that may lead an alias pattern. A pattern led by one of these
# is case-insensitive regardless of the rest: the vendor word is what papers
# spell inconsistently ("Nvidia", "nVidia"), and the model code following it
# removes any ambiguity a loose vendor match could otherwise introduce.
VENDOR_LEAD = {'nvidia', 'amd', 'ati', 'intel', 'google', 'apple', 'huawei',
               'aws', 'amazon', 'microsoft', 'meta', 'cambricon', 'baidu',
               'metax', 'biren', 'sunway', 'hygon', 'alibaba', 'pezy',
               'iluvatar', 'moore', 'nudt', 'habana', 'tesla', 'hgx'}


def vendor_qualified_case(pattern: str) -> str | None:
    """'insensitive' for a multi-token pattern led by a vendor name, else None.

    None means "leave it to registry `case: auto`" -- a bare code stays
    case-sensitive, which is what keeps `V100` off `v100` in a URL path.
    """
    tokens = pattern.split(' ')
    if len(tokens) < 2:
        return None
    lead = re.sub(r'\\(.)', r'\1', tokens[0]).lower()
    return 'insensitive' if lead in VENDOR_LEAD else None
