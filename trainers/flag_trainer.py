import torch
from tqdm import tqdm
from .base_trainer import BaseTrainer
from trainers import register_trainer

@register_trainer("flag")
class FlagTrainer(BaseTrainer):
    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--step-size', type=float, default=8e-3)
        parser.add_argument('-m', type=int, default=3)
        # fmt: on

    @staticmethod
    def train(model, device, loader, optimizer, args, calc_loss):
        model.train()

        loss_accum = 0
        for step, batch in enumerate(tqdm(loader, desc="Iteration")):
            batch = batch.to(device)


            if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
                pass
            else:
                optimizer.zero_grad()

                perturb = torch.FloatTensor(
                    batch.x.shape[0], args.emb_dim).uniform_(-args.step_size, args.step_size).to(device)
                perturb.requires_grad_()

                pred_list = model(batch, perturb)
                
                loss = calc_loss(pred_list, batch, args.m)

                for _ in range(args.m - 1):
                    loss.backward()
                    perturb_data = perturb.detach() + args.step_size * torch.sign(perturb.grad.detach())
                    perturb.data = perturb_data.data
                    perturb.grad[:] = 0

                    pred_list = model(batch, perturb)

                    loss = calc_loss(pred_list, batch, args.m)

                loss.backward()
                optimizer.step()

                loss_accum += loss.item()

        return loss_accum / (step + 1)
