import importlib
import os

from .base_trainer import BaseTrainer

TRAINER_REGISTRY = {}
TRAINER_CLASS_NAMES = set()

__all__ = {
    "BaseTrainer",
}


def get_trainer_and_parser(args, parser):
    trainer = TRAINER_REGISTRY[args.aug]
    trainer.add_args(parser)
    return trainer


def register_trainer(name, dataclass=None):
    """
    New tasks can be added to fairseq with the
    :func:`~fairseq.tasks.register_task` function decorator.
    For example::
        @register_task('classification')
        class ClassificationTask(FairseqTask):
            (...)
    .. note::
        All Tasks must implement the :class:`~fairseq.tasks.FairseqTask`
        interface.
    Args:
        name (str): the name of the task
    """

    def register_trainer_cls(cls):
        if name in TRAINER_REGISTRY:
            raise ValueError("Cannot register duplicate task ({})".format(name))
        if not issubclass(cls, BaseTrainer):
            raise ValueError("Trainer ({}: {}) must extend BaseTrainer".format(name, cls.__name__))
        if cls.__name__ in TRAINER_CLASS_NAMES:
            raise ValueError("Cannot register task with duplicate class name ({})".format(cls.__name__))
        TRAINER_REGISTRY[name] = cls
        TRAINER_CLASS_NAMES.add(cls.__name__)

        return cls

    return register_trainer_cls


# automatically import any Python files in the models/ directory
trainers_dir = os.path.dirname(__file__)
for file in os.listdir(trainers_dir):
    path = os.path.join(trainers_dir, file)
    if not file.startswith("_") and not file.startswith(".") and (file.endswith(".py") or os.path.isdir(path)):
        trainer_name = file[: file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module("trainers." + trainer_name)
