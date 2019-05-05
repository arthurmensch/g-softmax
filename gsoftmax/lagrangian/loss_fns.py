import torch


def pairwise_l2_dist(p):
    D = p.unsqueeze(2) - p.unsqueeze(1)
    return ((D * D).sum(-1) + 1E-10).sqrt()


def kernel_loss(K, predicted_weights, target_weights):
    weights_vec = torch.cat([predicted_weights, -target_weights], 1).unsqueeze(
        -1)
    embedding_loss = torch.matmul(weights_vec.transpose(1, 2),
                                  torch.matmul(K, weights_vec))
    return embedding_loss.squeeze()


def pairwise_l1_dist(p):
    D = p.unsqueeze(2) - p.unsqueeze(1)
    return D.abs().sum(-1)


def multiscale_l1_laplacian_loss(p_t, p_w, t_t, t_w, inv_scale_factors):
    D = pairwise_l1_dist(torch.cat([p_t, t_t], 1))
    losses = [kernel_loss(torch.exp(-D / sf), p_w, t_w) for sf in
              inv_scale_factors]  # this is likely imprecise ...
    return sum(losses)
