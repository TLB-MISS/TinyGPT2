import math

class AnnealingLR(object):
    """Anneals the learning rate."""

    def __init__(self, optimizer, local_rank, start_lr,
                 warmup_iter, plateau_iter, total_iters,
                 decay_style, last_iter, min_lr=0.0):

        # Class values.
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.min_lr = min_lr
        self.warmup_iter = warmup_iter
        self.plateau_iter = plateau_iter
        self.num_iters = last_iter
        self.end_iter = total_iters
        assert self.end_iter > 0
        self.decay_style = decay_style
        # Set the learning rate
        self.step(self.num_iters)

        if local_rank in [-1, 0]:
            print('Learning rate decay style: {}'.format(self.decay_style))

    def get_lr(self):
        """Learning rate decay functions from:
              https://openreview.net/pdf?id=BJYwwY9ll pg. 4"""

        # Warmup.
        if self.warmup_iter > 0 and self.num_iters <= self.warmup_iter:
            return float(self.start_lr) * (self.num_iters / self.warmup_iter)

        if self.decay_style == 'linear':
            lr = self.start_lr * ((self.end_iter - self.num_iters) / self.end_iter)
        elif self.decay_style == 'plateau':
            if self.num_iters <= self.plateau_iter:
                lr = self.start_lr
            else:
                lr = self.start_lr * (self.end_iter - self.num_iters) / (self.end_iter - self.plateau_iter)
        elif self.decay_style == 'cosine':
            lr = self.start_lr / 2.0 * (math.cos(
                math.pi * (self.num_iters / self.end_iter)) + 1)
        elif self.decay_style == 'exponential':
            # exp(-0.693) = 1/2
            lr = self.start_lr * math.exp(-0.693 * (self.num_iters / self.end_iter))
        else:
            lr = self.start_lr
        return max(lr, self.min_lr)

    def step(self, step_num=None):
        """Set lr for all parameters groups."""
        if step_num is None:
            step_num = self.num_iters + 1
        self.num_iters = step_num
        new_lr = self.get_lr()
        for group in self.optimizer.param_groups:
            group['lr'] = new_lr

    def state_dict(self):
        state_dict = {
            'num_iters': self.num_iters,
        }
        return state_dict

    def load_state_dict(self, sd):
        self.num_iters = sd['num_iters']
        self.step(self.num_iters)