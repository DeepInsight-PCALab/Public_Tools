class AnnealingPolicy:
    """
    method = 'fixed', 'step', 'exp', 'inv', 'poly', 'sigmoid'
    """
    def __init__(self,
        method,
        base_lr = 0.01,
        max_iter = 30000,
        gamma = 0.9,
        step = 1000,
        power = 1.1
        ):
        self.max_iter = float(max_iter)
        self.gamma = gamma
        self.step = float(step)
        self.power = power 
        self.base_lr = base_lr
        table = {'fixed'   : self.fixed,
                 'step'    : self.step_,
                 'exp'     : self.exp,
                 'inv'     : self.inv,
                 'poly'    : self.poly,
                 'sigmoid' : self.sigmoid}

        self.decay = table[method]

    def __call__(self, epoch):
        return self.decay(epoch)

    def fixed(self, epoch):
        return self.base_lr

    def step_(self, epoch):
        return self.base_lr * self.gamma**(np.floor(epoch / self.step))

    def exp(self, epoch):
        return self.base_lr * self.gamma**(epoch)

    def inv(self, epoch):
        return self.base_lr * (1 + self.gamma * epoch)**(-self.power)

    def poly(self, epoch):
        return self.base_lr * (1 - epoch / self.max_iter)**(self.power)

    def sigmoid(self, epoch):
        return self.base_lr * (1.0 / (1 + np.exp(-self.gamma * (epoch - self.step))))
