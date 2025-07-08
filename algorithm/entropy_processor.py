import torch
import torch.nn.functional as F
from transformers import LogitsProcessor


class AdaptiveEntropyLogitsProcessor(LogitsProcessor):
    """
    一个在线、自适应的Logits处理器，用于在LLM生成过程中动态调整搜索空间。

    它在每个生成步骤执行以下操作：
    1. 计算当前步骤的下一个Token预测的熵。
    2. 将熵与预设的阈值进行比较。
    3. 如果熵低（模型很自信），则应用一个较小的k值进行Top-K采样，积极缩小搜索空间。
    4. 如果熵高（模型不确定），则应用一个较大的k值，允许更多的探索。
    """

    def __init__(self, low_entropy_threshold: float, low_k: int = 3, high_k: int = 10):
        """
        初始化处理器。

        Args:
            low_entropy_threshold (float): 用于区分高/低不确定性的绝对熵阈值。
                                          这个值需要通过实验确定，它不再是[0,1]范围内的值。
            low_k (int): 当熵低时使用的k值（例如，3 或 5）。
            high_k (int): 当熵高时使用的k值（例如，20 或 50）。
        """
        if not (0 < low_k <= high_k):
            raise ValueError(f"k值必须满足 0 < low_k <= high_k，但得到 low_k={low_k}, high_k={high_k}")

        self.low_entropy_threshold = low_entropy_threshold
        self.low_k = low_k
        self.high_k = high_k

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        在每个生成步骤中被调用，以修改logits。

        Args:
            input_ids (torch.LongTensor): 到目前为止已生成的Token ID。
            scores (torch.FloatTensor): 当前步骤的Logits，形状为 (batch_size, vocab_size)。

        Returns:
            torch.FloatTensor: 修改后的Logits。
        """
        # 1. 从当前步的logits(scores)计算熵
        # 这是在线计算，只关心当前这一步
        probs = F.softmax(scores, dim=-1)
        log_probs = F.log_softmax(scores, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)  # 形状: (batch_size,)

        # 2. 对批次中的每个样本应用自适应逻辑
        for i in range(entropy.shape[0]):
            # 3. 比较绝对熵值和阈值，决定k的大小
            # 注意：这里不再需要归一化，直接使用原始熵值
            k = self.low_k if entropy[i] < self.low_entropy_threshold else self.high_k

            # 4. 应用Top-K逻辑，裁剪搜索空间
            indices_to_remove = scores[i].topk(k=scores.shape[-1] - k, sorted=False)[1]
            scores[i, indices_to_remove] = -float('inf')

        return scores
