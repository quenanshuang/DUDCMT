import torch
from torch import nn
from torch.nn.functional import one_hot
import torch.nn.functional as F


def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    # input_softmax = F.softmax(input_logits, dim=1)
    # target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_logits-target_logits)**2
    return mse_loss

###############lMFloss#####################################3
# class FocalLoss(nn.Module):
    
#     def __init__(self, alpha, gamma=2):
#         super().__init__()
#         self.alpha = alpha
#         self.gamma = gamma
    
#     def forward(self, output, target):
#         num_classes = output.size(1)
#         assert len(self.alpha) == num_classes, \
#             'Length of weight tensor must match the number of classes'
#         logp = F.cross_entropy(output, target, self.alpha)
#         p = torch.exp(-logp)
#         focal_loss = (1-p)**self.gamma*logp
 
#         return torch.mean(focal_loss)
class FocalLoss(nn.Module):
    """Focal loss function.
    Focal Loss for Dense Object Detection
    paper : https://arxiv.org/abs/1708.02002
    code from : https://github.com/pytorch/vision/blob/main/torchvision/ops/focal_loss.py

    Args:
        alpha (float, optional): The weight factor for the positive class. Defaults to 0.25.
        gamma (float, optional): The focusing parameter. Defaults to 2.
        reduction (str, optional): Specifies the reduction method for the loss. Defaults to "mean".

    Returns:
        torch.Tensor: The Focal loss.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2, reduction: str = "mean"):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculates the Focal loss.

        Args:
            inputs (torch.Tensor): A tensor of shape (B, C, H, W) representing the model's predictions.
            targets (torch.Tensor): A tensor of shape (B, C, H, W) representing the ground truth labels.

        Returns:
            torch.Tensor: The Focal loss.
        """
        # Calculate the binary cross-entropy loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
        # Calculate the probability based on the sigmoid function
        probs = torch.sigmoid(inputs)
        # Calculate the focal loss components
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * bce_loss

        # Reduce the loss if specified.
        if self.reduction == "mean":
            focal_loss = focal_loss.mean()
        elif self.reduction == "sum":
            focal_loss = focal_loss.sum()

        return focal_loss

class LDAMLoss(nn.Module):
    """LDAM loss function.
    Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss
    paper : https://arxiv.org/abs/1906.07413
    code from : https://github.com/kaidic/LDAM-DRW/blob/master/losses.py

    Args:
        max_m (float, optional): The maximum margin. Defaults to 0.5.
        s (float, optional): The scaling factor. Defaults to 30.

    Returns:
        torch.Tensor: The LDAM loss.
    """

    def __init__(self, max_m: float = 0.5, s: float = 30):
        super().__init__()
        self.cls_num_list = None
        self.m_list = None
        self.max_m = max_m
        self.s = s

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculates the LDAM loss.

        Args:
            inputs (torch.Tensor): A tensor of shape (B, C, H, W) representing the model's predictions.
            targets (torch.Tensor): A tensor of shape (B, C, H, W) representing the ground truth labels.

        Returns:
            torch.Tensor: The LDAM loss.
        """
        cls_num_list = self.calculate_batch_class_distribution(targets)
        m_list = self.calculate_class_margins(cls_num_list, self.max_m)


        margin = m_list[1] * targets.float()
        margin_adjusted_inputs = inputs - margin

        output = torch.where(targets == 1, margin_adjusted_inputs, inputs)

        return F.binary_cross_entropy_with_logits(self.s * output, targets.float())

    @staticmethod
    def calculate_batch_class_distribution(targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate the number of pixels for each class within a batch.

        Args:
        - targets (torch.Tensor): A tensor of shape (B, C, H, W)
          containing binary segmentation masks.

        Returns:
        - cls_num_list (torch.Tensor): A tensor with the number of pixels for each class within the batch.
        """
        # Assuming binary classification (0 for background, 1 for foreground)
        background_count = (targets == 0).sum().item()
        foreground_count = (targets == 1).sum().item()

        # Create the class number list tensor for the batch
        cls_num_list = torch.Tensor([background_count, foreground_count])
        
        return cls_num_list

    @staticmethod
    def calculate_class_margins(cls_num_list: torch.Tensor, max_m: float) -> torch.Tensor:
        """Calculates the class-dependent margins.

        Args:
            cls_num_list (torch.Tensor): A tensor containing the number of pixels for each class within the batch.
            max_m (float): The maximum margin.

        Returns:
            torch.Tensor: A tensor containing the class-dependent margins.
        """

        m_list = 1.0 / torch.sqrt(torch.sqrt(cls_num_list))
        m_list = m_list * (max_m / torch.max(m_list))

        return m_list

class LMFLoss(nn.Module):
    """LMF loss function.
    LMFLOSS: A Hybrid Loss For Imbalanced Medical Image Classification
    paper : https://arxiv.org/abs/2212.12741
    
    Args:
        max_m (float, optional): The maximum margin. Defaults to 0.5.
        s (float, optional): The scaling factor. Defaults to 30.
        alpha (float, optional): The weight factor for the positive class. Defaults to 1.
        gamma (float, optional): The focusing parameter. Defaults to 1.5.
        reduction (str, optional): Specifies the reduction method for the loss. Defaults to "mean".

    Returns:
        torch.Tensor: The LMF loss.
    """

    def __init__(self, max_m: float = 0.5, s: float = 30, alpha: float = 0.25, gamma: float = 1.5, reduction: str = "mean"):
        super().__init__()

        self.ldam = LDAMLoss(max_m, s)
        self.focal = FocalLoss(alpha, gamma, reduction)
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculates the LMF loss.

        Args:
            inputs (torch.Tensor): A tensor of shape (B, C, H, W) representing the model's predictions.
            targets (torch.Tensor): A tensor of shape (B, C, H, W) representing the ground truth labels.

        Returns:
            torch.Tensor: The LMF loss.
        """
        loss = self.ldam(inputs, targets) + self.focal(inputs, targets)

        return loss 

class DSCLoss(nn.Module):
    def __init__(self, num_classes=2, inter_weight=0.5, intra_weights=None, device='cuda'):
        super(DSCLoss, self).__init__()
        if intra_weights is not None:
            intra_weights = torch.tensor(intra_weights).to(device)
        self.ce_loss = nn.CrossEntropyLoss(weight=intra_weights)
        self.num_classes = num_classes
        self.intra_weights = intra_weights
        self.inter_weight = inter_weight
        self.device = device

    def dice_loss(self, prediction, target, weights=None):
        """Calculating the dice loss
        Args:
            prediction = predicted image
            target = Targeted image
        Output:
            dice_loss"""
        smooth = 1e-5

        prediction = torch.softmax(prediction, dim=1)
        batchsize = target.size(0)
        num_classes = target.size(1)
        prediction = prediction.view(batchsize, num_classes, -1)
        target = target.view(batchsize, num_classes, -1)

        intersection = (prediction * target)

        dice = (2. * intersection.sum(2) + smooth) / (prediction.sum(2) + target.sum(2) + smooth)
        dice_loss = 1 - dice.sum(0) / batchsize
        if weights is not None:
            weighted_dice_loss = dice_loss * weights
            return weighted_dice_loss.mean()
        return dice_loss.mean()

    def forward(self, pred, label):
        """Calculating the loss and metrics
            Args:
                prediction = predicted image
                target = Targeted image
                metrics = Metrics printed
                bce_weight = 0.5 (default)
            Output:
                loss : dice loss of the epoch """
        cel = self.ce_loss(pred, label)
        label_onehot = one_hot(label, num_classes=self.num_classes).permute(0, 3, 1, 2).contiguous()
        dicel = self.dice_loss(pred, label_onehot, self.intra_weights)
        loss = cel * (1 - self.inter_weight) + dicel * self.inter_weight
        return loss
