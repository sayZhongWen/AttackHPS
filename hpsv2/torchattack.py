import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .attack_base import Attack


class FGSM(Attack):
    r"""
    FGSM in the paper 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.FGSM(model, eps=8/255)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, eps=8 / 255):
        super().__init__("FGSM", model)
        self.eps = eps
        self.supported_mode = ["default", "targeted"]

    def forward(self, images, texts, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        texts = texts.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()

        images.requires_grad = True
        outputs = self.get_logits(images, texts)

        # Calculate loss
        if self.targeted:
            cost = -loss(outputs, target_labels)
        else:
            cost = loss(outputs, labels)

        # Update adversarial images
        grad = torch.autograd.grad(
            cost, images, retain_graph=False, create_graph=False
        )[0]

        adv_images = images + self.eps * grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        return adv_images
    
    
class PGD(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=10, random_start=True)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=10, random_start=True):
        super().__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.supported_mode = ["default", "targeted"]

    def forward(self, images, texts, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        texts = texts.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
            )
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.get_logits(adv_images, texts)

            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images



class PIFGSMPP(Attack):
    r"""
    Patch-wise++ Perturbation for Adversarial Targeted Attacks'
    [https://arxiv.org/abs/2012.15503]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        max_epsilon (float): maximum size of adversarial perturbation. (Default: 16/255)
        num_iter_set (float): number of iterations. (Default: 10)
        momentum (float): momentum. (Default: 1.0)
        amplification (float): to amplifythe step size. (Default: 10.0)
        prob (float): probability of using diverse inputs. (Default: 0.7)
        project_factor (float): To control the weight of project term. (Default: 0.8)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PIFGSMPP(model, eps=16/255, num_iter_set=10)
        >>> adv_images = attack(images, labels)

    """

    def __init__(
        self,
        model,
        max_epsilon=16 / 255,
        num_iter_set=10,
        momentum=1.0,
        amplification=10.0,
        prob=0.7,
        project_factor=0.8,
    ):
        super().__init__("PIFGSMPP", model)
        self.max_epsilon = max_epsilon
        self.num_iter_set = num_iter_set
        self.momentum = momentum
        self.amplification = amplification
        self.prob = prob
        self.project_factor = project_factor
        self.supported_mode = ["default", "targeted"]

    def forward(self, images, texts, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        texts = texts.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        images_min = self.clip_by_tensor(images - self.max_epsilon, t_min=0, t_max=1)
        images_max = self.clip_by_tensor(images + self.max_epsilon, t_min=0, t_max=1)
        adv_images = self.graph(images, texts, labels, images_min, images_max)

        return adv_images

    def clip_by_tensor(self, t, t_min, t_max):
        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
        return result

    def project_noise(self, images, P_kern, padding_size):
        images = F.conv2d(
            images, P_kern, padding=(padding_size, padding_size), groups=3
        )
        return images

    def gaussian_kern(self, kernlen=21, nsig=3):
        """Returns a 2D Gaussian kernel array."""
        import scipy.stats as st

        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        kernel = kernel.astype(np.float32)
        stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
        stack_kernel = np.expand_dims(stack_kernel, 1)
        stack_kernel = torch.tensor(stack_kernel).to(self.device)
        return stack_kernel

    def project_kern(self, kern_size):
        kern = np.ones((kern_size, kern_size), dtype=np.float32) / (kern_size ** 2 - 1)
        kern[kern_size // 2, kern_size // 2] = 0.0
        kern = kern.astype(np.float32)
        stack_kern = np.stack([kern, kern, kern])
        stack_kern = np.expand_dims(stack_kern, 1)
        stack_kern = torch.tensor(stack_kern).to(self.device)
        return stack_kern, kern_size // 2

    def graph(self, images, texts, labels, images_min, images_max):
        eps = self.max_epsilon
        num_iter = self.num_iter_set
        alpha = eps / num_iter
        alpha_beta = alpha * self.amplification
        gamma = alpha_beta * self.project_factor
        P_kern, padding_size = self.project_kern(3)
        T_kern = self.gaussian_kern(3, 3)

        images.requires_grad = True
        amplification = 0.0
        for _ in range(num_iter):
            # zero_gradients(images)
            if images.grad is not None:
                images.grad.detach_()
                images.grad.zero_()
            output_v3 = self.get_logits(images, texts)
            loss = F.cross_entropy(output_v3, labels)
            loss.backward()
            noise = images.grad.data
            noise = F.conv2d(
                noise, T_kern, padding=(padding_size, padding_size), groups=3
            )

            amplification += alpha_beta * torch.sign(noise)
            cut_noise = torch.clamp(abs(amplification) - eps, 0, 10000.0) * torch.sign(
                amplification
            )
            projection = gamma * torch.sign(
                self.project_noise(cut_noise, P_kern, padding_size)
            )

            if self.targeted:
                images = images - alpha_beta * torch.sign(noise) - projection
            else:
                images = images + alpha_beta * torch.sign(noise) + projection

            images = self.clip_by_tensor(images, images_min, images_max)
            images = images.detach().requires_grad_(True)

        return images.detach()