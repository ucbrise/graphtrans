from tqdm import tqdm
import torch

class BaseTrainer:

    @staticmethod
    def add_args(parser):
        pass
    
    @staticmethod
    def train(model, device, loader, optimizer, args, calc_loss):
        model.train()

        loss_accum = 0
        for step, batch in enumerate(tqdm(loader, desc="Train")):
            batch = batch.to(device)

            if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
                pass
            else:
                pred_list = model(batch)
                optimizer.zero_grad()

                loss = calc_loss(pred_list, batch)

                loss = loss / len(pred_list)

                loss.backward()
                optimizer.step()

                loss_accum += loss.item()

        print('Average training loss: {}'.format(loss_accum / (step + 1)))
        return loss_accum / (step + 1)
