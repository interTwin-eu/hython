from . import *


def metric_epoch(metric_func, y_pred, y_true, target_names):
    metrics = metric_func(y_pred, y_true, target_names)
    return metrics


def loss_batch(
    loss_func,
    output,
    target,
    opt=None,
    gradient_clip=None,
    model=None,
    add_losses: dict = {},
):
    if target.shape[-1] == 1:
        target = torch.squeeze(target)
        output = torch.squeeze(output)

    loss = loss_func(target, output)

    # compound more losses, in case dict is not empty
    # TODO: add user-defined weights
    for k in add_losses:
        loss += add_losses[k]

    if opt is not None:
        opt.zero_grad()
        loss.backward()

        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), **gradient_clip)

        opt.step()

    return loss


def train_val(
    trainer,
    model,
    train_loader,
    val_loader,
    epochs,
    optimizer,
    lr_scheduler,
    dp_weights,
    device,
):
    loss_history = {"train": [], "val": []}
    metric_history = {f"train_{target}": [] for target in trainer.P.target_names}
    metric_history.update({f"val_{target}": [] for target in trainer.P.target_names})

    best_loss = float("inf")

    epoch_iterator = tqdm(range(epochs)) if tqdm_support else range(epochs)

    for epoch in epoch_iterator:
        model.train()

        # set time indices for training
        # This has effect only if the trainer overload the method (i.e. for RNN)
        trainer.temporal_index([train_loader, val_loader])

        train_loss, train_metric = trainer.epoch_step(
            model, train_loader, device, opt=optimizer
        )

        model.eval()
        with torch.no_grad():
            # set time indices for validation
            # This has effect only if the trainer overload the method (i.e. for RNN)
            trainer.temporal_index([train_loader, val_loader])

            val_loss, val_metric = trainer.epoch_step(
                model, val_loader, device, opt=None
            )

        lr_scheduler.step(val_loss)

        loss_history["train"].append(train_loss)
        loss_history["val"].append(val_loss)

        for target in trainer.P.target_names:
            metric_history[f"train_{target}"].append(train_metric[target])
            metric_history[f"val_{target}"].append(val_metric[target])

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            trainer.save_weights(model, dp_weights)
            print("Copied best model weights!")

        if not tqdm_support:
            print(f"Epoch: {epoch}")
        print(f"Losses - train: {train_loss.item():.6f}  val: {val_loss.item():.6f}")

    model.load_state_dict(best_model_weights)

    return model, loss_history, metric_history
