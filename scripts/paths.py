"""Central file locations. Paths are resolved relative to the repository root,
so the scripts run unchanged on any machine once the input data are placed in data/."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

KOYAMA_DB = DATA / "Breakthrough Database.xlsx"      # Koyama et al. (2024), GitHub furtman/GAC_BV10_estimation
EPA_XLSX = DATA / "Phos_VOC_SciHubDataset.xlsx"      # Haupert et al. (2023), EPA ScienceHub, doi:10.23719/1529105
