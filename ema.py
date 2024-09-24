def update_average(model, ema_model, beta=0.999):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(beta).add_(param.data, alpha=(1 - beta))
