import torch

"""Imports from https://github.com/luisvalesilva/multisurv."""


class EmbraceNet(torch.nn.Module):
    """Embracement modality feature aggregation layer."""

    def __init__(self, device="cpu"):
        """Embracement modality feature aggregation layer.
        Note: EmbraceNet needs to deal with mini batch elements differently
        (check missing data and adjust sampling probailities accordingly). This
        way, we take the unusual measure of considering the batch dimension in
        every operation.
        Parameters
        ----------
        device: "torch.device" object
            Device to which input data is allocated (sampling index tensor is
            allocated to the same device).
        """
        super(EmbraceNet, self).__init__()
        self.device = device

    def _get_selection_probabilities(self, d, b):
        p = torch.ones(d.size(0), b)  # Size modalities x batch

        # Handle missing data
        for i, modality in enumerate(d):
            for j, batch_element in enumerate(modality):
                if len(torch.nonzero(batch_element)) < 1:
                    p[i, j] = 0

        # Equal chances to all available modalities in each mini batch element
        m_vector = torch.sum(p, dim=0)
        p /= m_vector

        return p

    def _get_sampling_indices(self, p, c, m):
        r = torch.multinomial(
            input=p.transpose(0, 1), num_samples=c, replacement=True
        )
        r = torch.nn.functional.one_hot(r.long(), num_classes=m)
        r = r.permute(2, 0, 1)

        return r

    def forward(self, x):
        m, b, c = x.size()

        p = self._get_selection_probabilities(x, b)
        r = self._get_sampling_indices(p, c, m).float().to(self.device)

        d_prime = r * x
        e = d_prime.sum(dim=0)

        return e


class Attention(torch.nn.Module):
    """Attention mechanism for multimodal representation fusion."""

    def __init__(self, size):
        """
        Parameters
        ----------
        size: int
            Attention vector size, corresponding to the feature representation
            vector size.
        """
        super(Attention, self).__init__()
        self.fc = torch.nn.Linear(size, size, bias=False)
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.Softmax(0)  # Across feature vector stack

    def _scale_for_missing_modalities(self, x, out):
        """Scale fused feature vector up according to missing data.
        If there were all-zero data modalities (missing/dropped data for
        patient), scale feature vector values up accordingly.
        """
        batch_dim = x.shape[1]
        for i in range(batch_dim):
            patient = x[:, i, :]
            zero_dims = 0
            for modality in patient:
                if modality.sum().data == 0:
                    zero_dims += 1

            if zero_dims > 0:
                scaler = zero_dims + 1
                out[i, :] = scaler * out[i, :]

        return out

    def forward(self, x):
        scores = self.tanh(self.fc(x))
        weights = self.softmax(scores)
        out = torch.sum(x * weights, dim=0)

        out = self._scale_for_missing_modalities(x, out)

        return out
