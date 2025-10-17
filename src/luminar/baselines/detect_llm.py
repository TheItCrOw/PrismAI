import torch

from luminar.baselines.log_likelihood import LogLikelihoodABC


class DetectLLM_LRR(LogLikelihoodABC):
    """DetectLLM Log-Likelihood Log-Rank Ratio Detector.

    Source:
        - Paper: https://aclanthology.org/2023.findings-emnlp.827.pdf
        - GitHub: https://github.com/mbzuai-nlp/DetectLLM
        - Implementation:
            - https://github.com/mbzuai-nlp/DetectLLM/blob/main/baselines/all_baselines.py#L35:L42
            - https://github.com/mbzuai-nlp/DetectLLM/blob/main/baselines/all_baselines.py#L94:L100
    """

    @torch.inference_mode()
    def _calculate_score(
        self,
        _probabilities: torch.Tensor,
        _log_probabilities: torch.Tensor,
        log_likelihoods: torch.Tensor,
        target_ranks: torch.Tensor,
        device: torch.device | None = None,
    ) -> float:
        """Implements the DetectLLM Log-Likelihood Log-Rank Ratio.

        Args:
            log_likelihoods (torch.Tensor): A tensor of log probabilities for each target token.
            target_ranks (torch.Tensor): A tensor of ranks for each target token.
            device (torch.device, optional): Device to run the calculations on. Defaults to None.

        Returns:
            float: The calculated log-likelihood log-rank ratio.
        """
        device = device or self.device
        return (
            -torch.div(
                log_likelihoods.to(device).sum(),
                target_ranks.to(device).log1p().sum(),
            )
            .cpu()
            .item()
        )
