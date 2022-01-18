import torch
from loguru import logger


def pad_batch(h_node, batch, max_input_len, get_mask=False):
    num_batch = batch[-1] + 1
    num_nodes = []
    masks = []
    for i in range(num_batch):
        mask = batch.eq(i)
        masks.append(mask)
        num_node = mask.sum()
        num_nodes.append(num_node)

    # logger.info(max(num_nodes))
    max_num_nodes = min(max(num_nodes), max_input_len)
    padded_h_node = h_node.data.new(max_num_nodes, num_batch, h_node.size(-1)).fill_(0)
    src_padding_mask = h_node.data.new(num_batch, max_num_nodes).fill_(0).bool()

    for i, mask in enumerate(masks):
        num_node = num_nodes[i]
        if num_node > max_num_nodes:
            num_node = max_num_nodes
        padded_h_node[-num_node:, i] = h_node[mask][-num_node:]
        src_padding_mask[i, : max_num_nodes - num_node] = True  # [b, s]

    if get_mask:
        return padded_h_node, src_padding_mask, num_nodes, masks, max_num_nodes
    return padded_h_node, src_padding_mask


def unpad_batch(padded_h_node, prev_h_node, num_nodes, origin_mask, max_num_nodes):
    """
    padded_h_node: [s, b, f]
    prev_h_node: [bxs, f]
    batch: [n]
    pad_mask: [b, s]
    """

    for i, mask in enumerate(origin_mask):
        num_node = num_nodes[i]
        if num_node > max_num_nodes:
            num_node = max_num_nodes
            # cutoff mask
            indices = mask.nonzero()
            indices = indices[-num_node:]
            mask = torch.zeros_like(mask)
            mask[indices] = True
        # logger.info("prev_h_node:", prev_h_node.size())
        # logger.info("padded_h_node:", padded_h_node.size())
        # logger.info("mask:", mask.size())
        prev_h_node = prev_h_node.masked_scatter(mask.unsqueeze(-1), padded_h_node[-num_node:, i])
    return prev_h_node
