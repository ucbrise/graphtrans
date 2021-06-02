from .code import CodeUtil
from .mol import MolUtil
from .ppa import PPAUtil
from .tud import TUUtil

DATASET_UTILS = {
    "ogbg-code": CodeUtil,
    "ogbg-code2": CodeUtil,
    "ogbg-molhiv": MolUtil,
    "ogbg-molpcba": MolUtil,
    "ogbg-ppa": PPAUtil,
    "NCI1": TUUtil,
    "NCI109": TUUtil,
    "DD": TUUtil,
}
