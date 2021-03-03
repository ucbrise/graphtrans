def code_loss_fn(_):
    multicls_criterion = torch.nn.CrossEntropyLoss()
    def calc_loss(pred_list, batch, m=1.0):
        loss = 0
        for i in range(len(pred_list)):
            loss += multicls_criterion(pred_list[i].to(torch.float32), batch.y_arr[:, i])
        loss = loss / len(pred_list)
        loss /= m
        return loss
    return calc_loss

def code_eval(model, device, loader, evaluator, arr_to_seq):
    model.eval()
    seq_ref_list = []
    seq_pred_list = []

    for step, batch in enumerate(tqdm(loader, desc="Eval")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred_list = model(batch)

            mat = []
            for i in range(len(pred_list)):
                mat.append(torch.argmax(pred_list[i], dim=1).view(-1, 1))
            mat = torch.cat(mat, dim=1)

            seq_pred = [arr_to_seq(arr) for arr in mat]

            # PyG >= 1.5.0
            seq_ref = [batch.y[i] for i in range(len(batch.y))]

            seq_ref_list.extend(seq_ref)
            seq_pred_list.extend(seq_pred)

    input_dict = {"seq_ref": seq_ref_list, "seq_pred": seq_pred_list}

    return evaluator.eval(input_dict)
