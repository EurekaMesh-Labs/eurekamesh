"""
Drug Discovery Constraints and Configuration.

Defines constraints, rules, and configuration for drug discovery
molecular generation campaigns.
"""

from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field


@dataclass
class DrugDiscoveryConstraints:
    """
    Constraints for drug discovery molecular generation.
    
    Attributes:
        target_protein: Target protein name/ID
        seed_molecules: Known hits/scaffolds to use as starting points
        desired_functional_groups: Functional groups to include/favor
        forbidden_functional_groups: Functional groups to avoid
        molecular_weight_range: (min, max) MW in Daltons
        logp_range: (min, max) partition coefficient
        hbd_max: Maximum hydrogen bond donors
        hba_max: Maximum hydrogen bond acceptors
        tpsa_range: (min, max) topological polar surface area
        sa_score_max: Maximum SA-Score (synthetic accessibility)
        allowed_atoms: Set of allowed atom symbols
        strategic_hypotheses: Free-text hypotheses to guide generation
    
    Examples:
        >>> constraints = DrugDiscoveryConstraints(
        ...     target_protein="EGFR kinase",
        ...     seed_molecules=["c1ccc2c(c1)nc(n2)N"],  # Aminobenzimidazole
        ...     desired_functional_groups=["amine", "amide"],
        ...     forbidden_functional_groups=["nitro", "azide"],
        ...     molecular_weight_range=(200, 500),
        ...     logp_range=(1, 5),
        ...     sa_score_max=3.5
        ... )
    """
    
    # Target information
    target_protein: str = ""
    target_description: Optional[str] = None
    
    # Seed molecules (starting points)
    seed_molecules: List[str] = field(default_factory=list)
    
    # Functional group constraints
    desired_functional_groups: List[str] = field(default_factory=list)
    forbidden_functional_groups: List[str] = field(default_factory=list)
    
    # Property windows
    molecular_weight_range: tuple = (150, 600)  # Daltons
    logp_range: tuple = (-2, 6)  # Partition coefficient
    hbd_max: int = 5  # H-bond donors (Lipinski)
    hba_max: int = 10  # H-bond acceptors (Lipinski)
    tpsa_range: tuple = (20, 140)  # Ų (topological polar surface area)
    rotatable_bonds_max: int = 10
    
    # Synthetic accessibility
    sa_score_max: float = 4.0  # Lower is better, <3.5 is good
    
    # Atom constraints
    allowed_atoms: Set[str] = field(default_factory=lambda: {
        'C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'B', 'Si'
    })
    
    # Strategic hypotheses (free-text guidance)
    strategic_hypotheses: List[str] = field(default_factory=list)
    
    # Diversity settings
    enforce_scaffold_diversity: bool = True
    target_scaffold_count: int = 20
    
    def to_prompt_context(self) -> str:
        """
        Convert constraints to LLM prompt context.
        
        Returns:
            Formatted string to inject in generation prompts
        """
        context_parts = []
        
        # Target
        if self.target_protein:
            context_parts.append(f"Target: {self.target_protein}")
            if self.target_description:
                context_parts.append(f"Description: {self.target_description}")
        
        # Seeds
        if self.seed_molecules:
            context_parts.append(
                f"\nSeed molecules (use as inspiration):\n" +
                "\n".join(f"  - {s}" for s in self.seed_molecules[:5])
            )
        
        # Functional groups
        if self.desired_functional_groups:
            context_parts.append(
                f"\nDesired functional groups: {', '.join(self.desired_functional_groups)}"
            )
        if self.forbidden_functional_groups:
            context_parts.append(
                f"Forbidden functional groups: {', '.join(self.forbidden_functional_groups)}"
            )
        
        # Property windows
        context_parts.append("\nProperty constraints:")
        context_parts.append(f"  - Molecular weight: {self.molecular_weight_range[0]}-{self.molecular_weight_range[1]} Da")
        context_parts.append(f"  - LogP: {self.logp_range[0]}-{self.logp_range[1]}")
        context_parts.append(f"  - H-bond donors: ≤{self.hbd_max}")
        context_parts.append(f"  - H-bond acceptors: ≤{self.hba_max}")
        context_parts.append(f"  - TPSA: {self.tpsa_range[0]}-{self.tpsa_range[1]} Ų")
        
        # Synthetic accessibility
        context_parts.append(f"  - SA-Score: ≤{self.sa_score_max} (must be synthesizable)")
        
        # Allowed atoms
        if self.allowed_atoms:
            context_parts.append(f"\nAllowed atoms: {', '.join(sorted(self.allowed_atoms))}")
        
        # Strategic hypotheses
        if self.strategic_hypotheses:
            context_parts.append("\nStrategic hypotheses:")
            for i, hyp in enumerate(self.strategic_hypotheses, 1):
                context_parts.append(f"  {i}. {hyp}")
        
        return "\n".join(context_parts)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'target_protein': self.target_protein,
            'target_description': self.target_description,
            'seed_molecules': self.seed_molecules,
            'desired_functional_groups': self.desired_functional_groups,
            'forbidden_functional_groups': self.forbidden_functional_groups,
            'molecular_weight_range': self.molecular_weight_range,
            'logp_range': self.logp_range,
            'hbd_max': self.hbd_max,
            'hba_max': self.hba_max,
            'tpsa_range': self.tpsa_range,
            'rotatable_bonds_max': self.rotatable_bonds_max,
            'sa_score_max': self.sa_score_max,
            'allowed_atoms': list(self.allowed_atoms),
            'strategic_hypotheses': self.strategic_hypotheses,
            'enforce_scaffold_diversity': self.enforce_scaffold_diversity,
            'target_scaffold_count': self.target_scaffold_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DrugDiscoveryConstraints':
        """Create from dictionary."""
        data = data.copy()
        if 'allowed_atoms' in data and isinstance(data['allowed_atoms'], list):
            data['allowed_atoms'] = set(data['allowed_atoms'])
        return cls(**data)


# Predefined constraint templates
KINASE_INHIBITOR_CONSTRAINTS = DrugDiscoveryConstraints(
    target_protein="Kinase (generic)",
    target_description="ATP-competitive kinase inhibitor",
    desired_functional_groups=["amine", "amide", "heterocycle"],
    forbidden_functional_groups=["nitro", "azide", "peroxide"],
    molecular_weight_range=(250, 550),
    logp_range=(1, 4),
    hbd_max=3,
    hba_max=7,
    tpsa_range=(40, 120),
    sa_score_max=3.5,
    strategic_hypotheses=[
        "Include H-bond acceptor for ATP pocket binding",
        "Explore 5- and 6-membered heterocycles as core scaffolds",
        "Consider moderate lipophilicity for membrane permeability"
    ]
)

GPCR_LIGAND_CONSTRAINTS = DrugDiscoveryConstraints(
    target_protein="GPCR (generic)",
    target_description="G-protein coupled receptor ligand",
    desired_functional_groups=["amine", "phenyl", "ether"],
    forbidden_functional_groups=["heavy_metals", "reactive_groups"],
    molecular_weight_range=(200, 450),
    logp_range=(2, 5),
    hbd_max=2,
    hba_max=6,
    tpsa_range=(30, 90),
    sa_score_max=3.0,
    strategic_hypotheses=[
        "Basic amine for receptor binding",
        "Lipophilic aromatic groups for hydrophobic pocket",
        "Compact structure for transmembrane access"
    ]
)

CNS_DRUG_CONSTRAINTS = DrugDiscoveryConstraints(
    target_protein="CNS target (generic)",
    target_description="Central nervous system penetrant drug",
    desired_functional_groups=["small_rings", "basic_amine"],
    forbidden_functional_groups=["large_polar_groups", "quaternary_ammonium"],
    molecular_weight_range=(200, 400),
    logp_range=(1, 3.5),
    hbd_max=2,
    hba_max=4,
    tpsa_range=(20, 60),  # Tight for BBB penetration
    sa_score_max=3.0,
    strategic_hypotheses=[
        "Small, lipophilic for blood-brain barrier penetration",
        "Limited H-bond donors/acceptors (TPSA < 60)",
        "Avoid substrates for P-glycoprotein efflux"
    ]
)


def get_constraint_template(template_name: str) -> DrugDiscoveryConstraints:
    """
    Get predefined constraint template.
    
    Args:
        template_name: One of "kinase", "gpcr", "cns"
    
    Returns:
        DrugDiscoveryConstraints instance
    
    Raises:
        ValueError: If template_name not recognized
    """
    templates = {
        'kinase': KINASE_INHIBITOR_CONSTRAINTS,
        'gpcr': GPCR_LIGAND_CONSTRAINTS,
        'cns': CNS_DRUG_CONSTRAINTS
    }
    
    template_name = template_name.lower()
    if template_name not in templates:
        raise ValueError(
            f"Unknown template '{template_name}'. "
            f"Available: {', '.join(templates.keys())}"
        )
    
    return templates[template_name]

