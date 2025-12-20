import pytest
from rdkit import Chem
from eurekamesh.domains.drug_discovery import SMILESCanonicalizer

@pytest.fixture(scope="module")
def canon():
    return SMILESCanonicalizer()

@pytest.mark.parametrize("pair", [
    ("CCO", "OCC"),            # ethanol variants
    ("c1ccccc1", "C1=CC=CC=C1") # benzene
])
def test_canonical_equivalence(canon, pair):
    a, b = pair
    ca = canon.canonicalize(a)
    cb = canon.canonicalize(b)
    assert ca == cb

@pytest.mark.parametrize("smiles,valid", [
    ("CCO", True),
    ("C1=CC=CC=C1", True),
    ("CCOC(=O)C", True),
    ("CCOC(", False),          # broken paren
    ("C(C", False)             # unbalanced paren
])
def test_is_valid(canon, smiles, valid):
    assert canon.is_valid(smiles) == valid
