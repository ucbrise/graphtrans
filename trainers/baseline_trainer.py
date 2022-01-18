from trainers import register_trainer

from .base_trainer import BaseTrainer


@register_trainer("baseline")
class BaselineTrainer(BaseTrainer):
    @staticmethod
    def name(args):
        return "baseline"
