from collections.abc import Callable
from typing import Any, Union, Literal

import hydra
import torch
from omegaconf import DictConfig, ListConfig
from torch import nn
import logging


__all__ = ["MARVEL"]


from typing import Callable, Literal, cast
import torch


def _get_reduction_function(reduction: Literal["mean", "sum", "none", None]) -> Callable:
    """
    Get the reduction function for the loss given the reduction string.

    Args:
        reduction (Literal["mean", "sum", "none", None]): String specifying the reduction function.

    """
    if reduction is not None:
        reduction_converted = reduction.lower().strip()
        if reduction_converted not in ["mean", "sum", "none"]:
            raise ValueError(f"Invalid reduction: {reduction}")
    else:
        reduction_converted = None

    if reduction_converted == "mean":
        return torch.mean
    elif reduction_converted == "sum":
        return torch.sum
    elif reduction_converted == "none" or reduction_converted is None:
        return lambda x: x
    else:
        # This block should never be reached due to the validation above
        raise NotImplementedError(f"Reduction <{reduction_converted}> is not implemented.")


class MARVEL(nn.Module):
    """Loss function for training ensemble models.

    The loss can be combined as a sum of the first moment, the second moment and randomized moments.

    Args:
        func (str or ListConfig[str]): Function to be applied to the inputs. Default is the identity "lambda x: x".
        centralize (bool): If True, the mean is subtracted before the function is applied. Default is True.
        random_power_min (float): Minimum power for the random function (p). Default is 1.
        random_power_max (float): Maximum power for the random function (p). Default is 3.
        use_first_loss_term (bool): If True, the first loss term (mse) is used. Default is True.
        use_random_loss_term (bool): If True, the random loss term is used. Default is True.
        normalization (Callable): Function to normalize the loss terms (e.g. CustomNormalizer). Default is None.
        return_random_terms (bool): If True, the random loss terms are added to the returned dict. Default is False.
        n_p_values_per_sample (int): Number of p values to be sampled per input-output pair. Default is 1.
        random_power_warmup_steps (int): Number of steps to linearly increase the maximum p from 1 to
            random_power_max. Default is 0.
        random_loss_weight (float): Weighting factor for the random loss term. Default is 1.
    """

    def __init__(
        self,
        func: str | ListConfig | list[str] = "lambda x, p: x",
        centralize: bool = True,
        random_power_min: float = 1.0,
        random_power_max: float = 3.0,
        use_first_loss_term: bool = True,
        use_random_loss_term: bool = True,
        normalization: Callable = None,
        return_random_terms: bool = False,
        n_p_values_per_sample: int = 1,
        random_power_warmup_steps: int = 0,
        random_loss_weight: float = 1.0,
    ):

        super().__init__()
        self.console_logger = logging.getLogger(__name__)
        self.centralize = centralize
        self._random_power_max = random_power_max
        self.min_p = random_power_min
        self.use_first_loss_term = use_first_loss_term
        self.use_random_loss_term = use_random_loss_term
        self.normalization = self._setup_normalization(normalization)
        self.func = self._setup_function(func)
        self.return_random_terms = return_random_terms
        self.n_p_values_per_sample = n_p_values_per_sample
        self.random_power_warmup_steps = random_power_warmup_steps
        self.random_loss_weight = random_loss_weight

        self.device = "cpu"
        self.random_power_max = self._random_power_max if random_power_warmup_steps == 0 else 1
        self.call_counter = 0

    @staticmethod
    def _setup_train_data_for_normalization(data: torch.Tensor) -> torch.Tensor | None:
        if data is None:
            return

        flat_data = data.flatten(start_dim=1)
        return flat_data

    def _set_random_power(self):
        """Method for increasing the random power from min_p to random_power_max linearly."""
        if self.random_power_warmup_steps == 0:
            return
        self.random_power_max = min(
            self._random_power_max,
            ((self._random_power_max - self.min_p) * self.call_counter / self.random_power_warmup_steps) + self.min_p,
        )
        self.call_counter += 1

    @staticmethod
    def _setup_normalization(normalization) -> Callable or None:
        if normalization is None:
            return None
        if isinstance(normalization, Callable):
            return normalization
        if isinstance(normalization, Union[list, ListConfig, DictConfig]):
            return hydra.utils.instantiate(normalization)

        raise ValueError(
            f"<normalization> has to be a None, callable or hydra-config. But you " f"provided <{normalization}>"
        )

    def _setup_function(self, func: str | ListConfig | list[str]) -> list[Callable]:
        if not isinstance(func, str):
            _func = [eval(compile(func_str, "<string>", "eval")) for func_str in func]
        else:
            _func = eval(compile(func, "<string>", "eval"))

        if not isinstance(_func, Union[list, ListConfig]):
            _func = [_func]

        # testing if the provided function is callable with the necessary format:
        try:
            for function in _func:
                self.console_logger.debug(f"Testing function <{function}>.")
                function(torch.ones(size=(1, 1)), torch.ones(size=(1, 1)))
        except Exception as exc:
            raise ValueError("The provided function is not valid. Does it accept two arguments?") from exc
        return _func

    def forward(self, ensemble_inputs: torch.Tensor, targets: torch.Tensor) -> dict[str : torch.Tensor]:
        """nn.Module forward method. Calculates the loss for the given inputs and targets.

        Args:
            ensemble_inputs (torch.Tensor): Input of shape (ensemble_size, batch_size, *).
            targets (torch.Tensor): Target of shape (batch_size, *). Since the targets do not come in ensemble form,
                the target does not have an ensemble dimension.

        Returns:
            dict[str: torch.Tensor]: Dictionary containing the loss terms. The keys are "first_moment", "second_moment",
                "random_moment", "loss". "loss" is the sum of the other terms.
        """
        self.device = ensemble_inputs.device

        first_moment = None
        if self.use_first_loss_term:
            first_moment = torch.nn.functional.mse_loss(ensemble_inputs.mean(dim=0), targets)

        random_moment = torch.zeros(1, device=self.device)

        # calculating the random moment for multiple p values per sample
        for _ in range(self.n_p_values_per_sample):
            # if there are multiple random functions we sum them all up here:
            for func_num, function in enumerate(self.func):
                this_random_moment, random_moment_term_1, random_moment_term_2 = self._calculate_random_moment(
                    ensemble_inputs, targets, func=function, func_num=func_num
                )
                random_moment += this_random_moment

        random_moment /= self.n_p_values_per_sample
        random_moment *= self.random_loss_weight

        loss = torch.zeros(1, device=self.device)
        if self.use_first_loss_term:
            loss += first_moment

        if self.use_random_loss_term:
            loss += random_moment

        # check if the loss is nan and dump a complete traceback for all vars for debugging:
        if not torch.isfinite(loss):
            raise RuntimeError("Loss is nan.")

        return_dict = dict(
            first_moment=first_moment.detach() if first_moment else torch.zeros(1, device=self.device),
            random_moment=random_moment.detach() if random_moment else torch.zeros(1, device=self.device),
            loss=loss,
        )

        if self.return_random_terms:
            return_dict["random_moment_term_1"] = random_moment_term_1.detach().cpu()
            return_dict["random_moment_term_2"] = random_moment_term_2.detach().cpu()

        if loss.requires_grad:
            # only update the random power during training:
            self._set_random_power()
        return return_dict

    def _calculate_random_moment(
        self,
        ensemble_inputs: torch.Tensor,
        targets: torch.Tensor,
        func: Callable,
        random_power: float | torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """Wrapper for _calculate_moment to calculate the random moment.

        Args:
            ensemble_inputs (torch.Tensor): Input of shape (ensemble_size, batch_size, *).
            targets (torch.Tensor): Target of shape (batch_size, *). Since the targets do not come in ensemble form,
                the target does not have an ensemble dimension.
            func (Callable): Function to be applied to the inputs.
            random_power (float or torch.Tensor): Power (p) for the function. Default is None, which means that
                a random power is sampled from a uniform distribution between 1 and self.random_power_max.
            **kwargs: Arguments that are passed to _calculate_moment.

        Returns:
            torch.Tensor: Loss term when using randomly drawn p-values. Size is (1)
        """
        batch_size = targets.shape[0]

        if random_power is None:
            random_power = (
                torch.rand(size=[batch_size], device=self.device) * (self.random_power_max - self.min_p)
            ) + self.min_p
        elif isinstance(random_power, torch.Tensor):
            random_power = random_power.to(self.device)
        else:
            random_power = torch.ones(size=[batch_size], device=self.device) * random_power

        min_length = max(len(ensemble_inputs.shape), 2)
        while len(random_power.shape) < min_length:
            random_power = random_power.unsqueeze(dim=-1)

        return self._calculate_moment(ensemble_inputs, targets, func=func, power=random_power, **kwargs)

    def _calculate_moment(
        self,
        ensemble_inputs: torch.Tensor,
        targets: torch.Tensor,
        func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        power: torch.Tensor | float = 1,
        reduction: str = "mean",
        func_num: int = 0,
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """Method to calculate a moment-like term for the loss. The function_str mainly determines the output.

        Args:
            ensemble_inputs (torch.Tensor): Input of shape (ensemble_size, batch_size, *).
            targets (torch.Tensor): Target of shape (batch_size, *). Since the targets do not come in ensemble form,
                the targets do not have an ensemble dimension.
            func (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): Function to be applied to the inputs. It will
                be invoked by pythons eval function and should accept two arguments. The first argument is the input
                and the second argument is the power. The function should return a torch.Tensor.
            power (Union[torch.Tensor, float, None]): p-value to replace the placeholder in the function_str. Has to be
                broadcastable to the shape of the targets. Default is 1.
            reduction (str): Reduction method for the loss. Default is "mean". If "none", the loss is not reduced and
                has an entry for each sample in the batch.
            func_num (int): In case that self.loss_func is a list, this is the index of the function to be
                used.

        Returns:
            torch.Tensor: Loss term of shape (1).
            torch.Tensor: First term of the loss term of shape (batch_size, 1, ...). (|y| +c)^p
            torch.Tensor: Second term of the loss term of shape (batch_size, 1, ...). E_z[|g(z, u)| + c^p]
        """

        self.check_tensor_shapes(ensemble_inputs, targets)

        targets_reshaped = targets.unsqueeze(1)  # targets_reshaped has shape (batch_size, 1, -1)

        # convert the power to a tensor if it is a float:
        if isinstance(power, Union[float, int]):
            power = torch.ones(size=[1], device=self.device) * power

        # check if the power is broadcastable to the targets. Afterwards power has same number of dimensions as targets,
        # but of shape (batch_size, 1, ...) with ... meaning that the dimensions are filled with 1s.
        if power.shape != targets_reshaped.shape:
            try:
                power = power.expand(targets_reshaped.shape[0], *[1 for _ in range(len(targets_reshaped.shape) - 1)])
            except RuntimeError as err:
                raise RuntimeError(
                    f"Power shape {power.shape} can not be broadcast to the target shape {targets_reshaped.shape}."
                ) from err
        power = power.to(self.device)
        if self.centralize:
            ensemble_mean = torch.mean(ensemble_inputs, dim=0, keepdim=True)  # shape (1, batch_size, -1)
            ensemble_inputs = ensemble_inputs - ensemble_mean
            targets_reshaped = targets_reshaped - torch.swapaxes(ensemble_mean, 1, 0)

        inputs_power = power.clone()
        while len(inputs_power.shape) < len(ensemble_inputs.shape):
            inputs_power = inputs_power.unsqueeze(dim=-1)
        transformed_inputs = torch.mean(func(ensemble_inputs.swapaxes(0, 1), inputs_power), dim=1)
        transformed_targets = func(targets_reshaped, power)

        # if transferred_inputs is missing a dimension, add it:
        if len(transformed_inputs.shape) < len(transformed_targets.shape):
            transformed_inputs = transformed_inputs.unsqueeze(dim=1)

        # to handle possible function outputs that return complex numbers, we take the absolute value:
        distances = torch.abs(self.l2_norm(transformed_inputs, transformed_targets))

        if self.normalization is not None:
            distances = self.normalization(distances, power.flatten(), func_num=func_num)

        reduction = _get_reduction_function(reduction)
        loss = reduction(distances)

        if not torch.isfinite(loss):
            # check if debug mode is on:
            if self.console_logger.level == 10:
                # save inputs and targets to disk:
                torch.save(ensemble_inputs, "ensemble_inputs.pt")
                torch.save(targets, "targets.pt")
                torch.save(power, "power.pt")
            self.console_logger.error(
                f"Loss is not finite. Loss: {loss} | "
                f"max_power: {power.max()} | "
                f"min_power: {power.min()} | "
                f"max_inputs: {ensemble_inputs.max()} | "
                f"min_inputs: {ensemble_inputs.min()} | "
                f"max_targets: {targets.max()} | "
                f"min_targets: {targets.min()} | "
                f"max_transformed_inputs: {transformed_inputs.max()} | "
                f"min_transformed_inputs: {transformed_inputs.min()} | "
                f"max_transformed_targets: {transformed_targets.max()} | "
                f"min_transformed_targets: {transformed_targets.min()}"
            )
            raise ValueError("Loss is not finite.")

        return loss, transformed_targets.detach(), transformed_inputs.detach()

    @staticmethod
    def l2_norm(input_tensor: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
        """Calculates the squared l2 norm between input_tensor and target_tensor.

        Args:
            input_tensor (torch.tensor): Input tensor of shape (batch_size, *).
            target_tensor (torch.tensor): Target tensor of shape (batch_size, *). Has to have the same shape as
                input_tensor.

        Returns:
            torch.tensor: l2 norm of shape (batch_size, 1).
        """

        input_tensor = input_tensor.flatten(start_dim=1)
        target_tensor = target_tensor.flatten(start_dim=1)
        l2 = (input_tensor - target_tensor).pow(2).mean(dim=1)
        return l2

    @staticmethod
    def check_tensor_shapes(input_tensor: torch.Tensor, target_tensor: torch.Tensor):
        target_shape = target_tensor.shape

        if input_tensor[0].shape != target_shape:
            raise ValueError(
                f"The ensemble inputs and targets must have different shapes, because the targets do not "
                f"have an ensemble dimension. The ensemble dimension of the inputs has to be the 1st "
                f"dimension. Got shapes ensemble_input: <{input_tensor.shape}> and "
                f"targets: <{target_tensor.shape}>."
            )

    def __str__(self):
        return "GNNLoss"
