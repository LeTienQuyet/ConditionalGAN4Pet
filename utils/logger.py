def training_setting(num_epochs, batch_size, num_dims, device, gen_params, dis_params, lr, betas, width=80):
    line = "-" * width
    print(line)
    print(f"|{'Details of Training':^{width-2}}|")
    print(line)

    # Setting
    setting_str = f"Epochs = {num_epochs}, Batch size = {batch_size}, Latent dims = {num_dims}, Device = {device}"
    print(f"|{setting_str:^{width-2}}|")
    print(line)

    # Params
    params_str = f"Generator params = {gen_params:,} Discriminator params = {dis_params:,}"
    print(f"|{params_str:^{width-2}}|")
    print(line)

    # Optimizer
    optimizer_str = f"Learning rate = {lr}, betas = {betas}"
    print(f"|{optimizer_str:^{width-2}}|")
    print(line)

def epoch_summary(epoch, fid_score, total_lossGen, total_lossDis, width=80):
    line = "-" * width
    # Epoch
    print(line)
    epoch_str = f"Epoch = {epoch}"
    print(f"|{epoch_str:^{width - 2}}|")
    print(line)

    # Train loss
    loss_str = f"Loss → Generator: {total_lossGen:.4f}     Discriminator: {total_lossDis:.4f}"
    print(f"|{loss_str:^{width - 2}}|")
    print(line)

    # Validation metrics
    metric_str = f"Metrics → FID = {fid_score:.4f}"
    print(f"|{metric_str:^{width - 2}}|")
    print(line)