import torch


def torch_mean_square_error(
    y_pred: torch.Tensor,
    y_real: torch.Tensor,
    reduction: str = "mean",
    flatten_start_dim=1,
) -> torch.Tensor:
    """
    Compute mean square error between y_pred(tensor) and y_real(tensor)

    MSE error can be calcualted as following equation:

        MSE(x, y) = (x - y) ^ 2

    if x and y are matrixs, MSE error over matrix should be the mean value of MSE error over all elements.

        MSE(X, Y) = mean((X - Y) ^ 2)

    By this equation, we can easily tell that MSE is an symmtrical measurement:
        MSE(X, Y) == MSE(Y, X)
        MSE(0, X) == X ^ 2

    Args:
        y_pred (torch.Tensor): _description_
        y_real (torch.Tensor): _description_
        reduction (str, optional): _description_. Defaults to 'mean'.

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        torch.Tensor: _description_
    """
    if y_pred.shape != y_real.shape:
        raise ValueError(
            f"Can not compute mse loss for tensors with different shape. "
            f"({y_pred.shape} and {y_real.shape})"
        )
    reduction = str(reduction).lower()

    if y_pred.ndim == 1:
        y_pred = y_pred.unsqueeze(1)
        y_real = y_real.unsqueeze(1)

    diff = torch.pow(y_pred.float() - y_real.float(), 2).flatten(start_dim=flatten_start_dim)
    mse = torch.mean(diff, dim=-1)

    if reduction == "mean":
        return torch.mean(mse)
    elif reduction == "sum":
        return torch.sum(mse)
    elif reduction == "none":
        return mse
    elif reduction == "max":
        return torch.max(mse)
    else:
        raise ValueError(f"Unsupported reduction method.")


def torch_snr_error(
    y_pred: torch.Tensor,
    y_real: torch.Tensor,
    reduction: str = "mean",
    flatten_start_dim=1,
) -> torch.Tensor:
    """
    Compute SNR between y_pred(tensor) and y_real(tensor)

    SNR can be calcualted as following equation:

        SNR(pred, real) = (pred - real) ^ 2 / (real) ^ 2

    if x and y are matrixs, SNR error over matrix should be the mean value of SNR error over all elements.

        SNR(pred, real) = mean((pred - real) ^ 2 / (real) ^ 2)

    Args:
        y_pred (torch.Tensor): _description_
        y_real (torch.Tensor): _description_
        reduction (str, optional): _description_. Defaults to 'mean'.

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        torch.Tensor: _description_
    """
    if y_pred.shape != y_real.shape:
        raise ValueError(
            f"Can not compute snr loss for tensors with different shape. "
            f"({y_pred.shape} and {y_real.shape})"
        )
    reduction = str(reduction).lower()

    if y_pred.ndim == 1:
        y_pred = y_pred.unsqueeze(1)
        y_real = y_real.unsqueeze(1)

    y_pred = y_pred.flatten(start_dim=flatten_start_dim).float()
    y_real = y_real.flatten(start_dim=flatten_start_dim).float()

    noise_power = torch.pow(y_pred - y_real, 2).sum(dim=-1)
    signal_power = torch.pow(y_real, 2).sum(dim=-1)
    snr = (noise_power) / (signal_power + 1e-7)

    if reduction == "mean":
        return torch.mean(snr)
    elif reduction == "sum":
        return torch.sum(snr)
    elif reduction == "none":
        return snr
    elif reduction == "max":
        return torch.max(snr)
    else:
        raise ValueError(f"Unsupported reduction method.")

def torch_rel_diff_error(
    y_pred: torch.Tensor,
    y_real: torch.Tensor,
    reduction: str = "mean",
    flatten_start_dim=1,
) -> torch.Tensor:
    """
    Compute SNR between y_pred(tensor) and y_real(tensor)

    SNR can be calcualted as following equation:

        SNR(pred, real) = (pred - real) ^ 2 / (real) ^ 2

    if x and y are matrixs, SNR error over matrix should be the mean value of SNR error over all elements.

        SNR(pred, real) = mean((pred - real) ^ 2 / (real) ^ 2)

    Args:
        y_pred (torch.Tensor): _description_
        y_real (torch.Tensor): _description_
        reduction (str, optional): _description_. Defaults to 'mean'.

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        torch.Tensor: _description_
    """
    if y_pred.shape != y_real.shape:
        raise ValueError(
            f"Can not compute rel diff loss for tensors with different shape. "
            f"({y_pred.shape} and {y_real.shape})"
        )
    reduction = str(reduction).lower()

    if y_pred.ndim == 1:
        y_pred = y_pred.unsqueeze(0)
        y_real = y_real.unsqueeze(0)

    y_pred = y_pred.flatten(start_dim=flatten_start_dim).float()
    y_real = y_real.flatten(start_dim=flatten_start_dim).float()

    noise_power = torch.abs(y_pred - y_real)
    signal_power = torch.abs(y_real)
    diff = torch.quantile(((noise_power) / (signal_power + 1e-9)).abs(), 0.999, dim=-1)

    if reduction == "mean":
        return torch.mean(diff)
    elif reduction == "sum":
        return torch.sum(diff)
    elif reduction == "none":
        return diff
    elif reduction == "max":
        return torch.max(diff)
    else:
        raise ValueError(f"Unsupported reduction method.")
    
