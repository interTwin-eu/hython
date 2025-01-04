from . import *


def train_val(
    trainer,
    model,
    train_loader,
    val_loader,
    epochs,
    device,
):
    loss_history = {"train": [], "val": []}
    metric_history = {f"train_{target}": [] for target in trainer.cfg.target_variables}
    metric_history.update(
        {f"val_{target}": [] for target in trainer.cfg.target_variables}
    )

    best_loss = float("inf")

    epoch_iterator = tqdm(range(epochs)) if tqdm_support else range(epochs)

    trainer.init_trainer(model)

    for epoch in epoch_iterator:
        train_loss, train_metric, val_loss, val_metric = trainer.train_valid_epoch(
            model, train_loader, val_loader, device
        )

        trainer.lr_scheduler.step(val_loss)

        loss_history["train"].append(train_loss)
        loss_history["val"].append(val_loss)

        for target in trainer.cfg.target_variables:
            metric_history[f"train_{target}"].append(train_metric[target])
            metric_history[f"val_{target}"].append(val_metric[target])

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            trainer.save_weights(model)
            print("Copied best model weights!")

        if not tqdm_support:
            print(f"Epoch: {epoch}")
        print(f"Losses - train: {train_loss.item():.6f}  val: {val_loss.item():.6f}")

    model.load_state_dict(best_model_weights)

    return model, loss_history, metric_history
