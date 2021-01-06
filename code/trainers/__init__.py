from .base_trainer import BaseTrainer
from .flag_trainer import FlagTrainer

TRAINER_REGISTRY = {}
TRAINER_CLASS_NAMES = set()

__all__ = {
    "BaseTrainer",
    "FlagTrainer"
}

def setup_trainer(args, parser):
    return TRAINER_REGISTRY[args.aug_method].setup_trainer(args, parser)


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
            raise ValueError(
                "Trainer ({}: {}) must extend BaseTrainer".format(name, cls.__name__)
            )
        if cls.__name__ in TRAINER_CLASS_NAMES:
            raise ValueError(
                "Cannot register task with duplicate class name ({})".format(
                    cls.__name__
                )
            )
        TRAINER_REGISTRY[name] = cls
        TRAINER_CLASS_NAMES.add(cls.__name__)

        return cls

    return register_trainer_cls