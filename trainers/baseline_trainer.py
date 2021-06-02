from .base_trainer import BaseTrainer
from trainers import register_trainer


@register_trainer("baseline")
class BaselineTrainer(BaseTrainer):
    @staticmethod
    def name(args):
        return "baseline"
