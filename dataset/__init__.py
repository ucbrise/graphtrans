from .code import CodeUtil
from .mol import MolUtil
from .ppa import PPAUtil

DATASET_UTILS = {
    'ogbg-code': CodeUtil,
    'ogbg-code2': CodeUtil,
    'ogbg-molhiv': MolUtil,
    'ogbg-molpcba': MolUtil,
    'ogbg-ppa': PPAUtil,
}