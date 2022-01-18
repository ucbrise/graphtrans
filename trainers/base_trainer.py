import torch
import wandb
from loguru import logger
from tqdm import tqdm


class BaseTrainer:
    @staticmethod
    def transform(args):
        return None

    @staticmethod
    def add_args(parser):
        pass

    @staticmethod
    def train(model, device, loader, optimizer, args, calc_loss, scheduler=None):
        model.train()

        loss_accum = 0
        t = tqdm(loader, desc="Train")
        for step, batch in enumerate(t):
            batch = batch.to(device)

            if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
                pass
            else:
                optimizer.zero_grad()
                pred_list = model(batch)

                loss = calc_loss(pred_list, batch)

                loss.backward()
                if args.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

                if scheduler:
                    scheduler.step()

                detached_loss = loss.item()
                loss_accum += detached_loss
                t.set_description(f"Train (loss = {detached_loss:.4f}, smoothed = {loss_accum / (step + 1):.4f})")
                wandb.log({"train/iter-loss": detached_loss, "train/iter-loss-smoothed": loss_accum / (step + 1)})

        logger.info("Average training loss: {:.4f}".format(loss_accum / (step + 1)))
        return loss_accum / (step + 1)

    @staticmethod
    def name(args):
        raise NotImplemented
