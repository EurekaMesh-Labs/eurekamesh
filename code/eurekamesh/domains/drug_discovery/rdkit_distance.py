from typing import Any, List
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from ...core.interfaces import DistanceMetric


class RdkitECFPDistance(DistanceMetric):
    """
    Distance metric using RDKit ECFP4 fingerprints and Tanimoto similarity.
    Returns distance in [0,1] as (1 - tanimoto_similarity).
    Accepts SMILES strings as inputs.
    """
    def __init__(self, radius: int = 2, n_bits: int = 2048):
        self.radius = radius
        self.n_bits = n_bits

    def _fp(self, smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits)

    def distance(self, obj1: Any, obj2: Any) -> float:
        fp1 = self._fp(str(obj1))
        fp2 = self._fp(str(obj2))
        if fp1 is None or fp2 is None:
            # If cannot parse, treat as maximally different
            return 1.0
        sim = DataStructs.TanimotoSimilarity(fp1, fp2)
        return 1.0 - float(sim)

    def batch_distance(self, obj: Any, obj_list: List[Any]) -> List[float]:
        fpq = self._fp(str(obj))
        if fpq is None:
            return [1.0] * len(obj_list)
        fps = []
        for o in obj_list:
            fp = self._fp(str(o))
            fps.append(fp)
        # Compute similarities where possible
        sims = []
        for fp in fps:
            if fp is None:
                sims.append(0.0)
            else:
                sims.append(DataStructs.TanimotoSimilarity(fpq, fp))
        return [1.0 - float(s) for s in sims]



