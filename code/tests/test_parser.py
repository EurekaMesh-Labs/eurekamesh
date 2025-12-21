from experiments.parser import parse_llm_smiles_output

def test_parse_json_array():
    text = '["CCO", "C1=CC=CC=C1"]'
    out = parse_llm_smiles_output(text)
    assert out == ["CCO", "C1=CC=CC=C1"]

def test_parse_lines_with_noise():
    text = """
    You asked:
    1. CCO
    2. C1=CC=CC=C1
    - CCOC(=O)C
    """
    out = parse_llm_smiles_output(text)
    assert "CCO" in out and "C1=CC=CC=C1" in out and "CCOC(=O)C" in out

