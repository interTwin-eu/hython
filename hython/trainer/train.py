from . import *
from hython.models import ModelLogAPI

def train_val(
    trainer,
    model,
    train_loader,
    val_loader,
    device,
    cfg
):
    loss_history = {"train": [], "val": []}
    metric_history = {f"train_{target}": [] for target in trainer.cfg.target_variables}
    metric_history.update(
        {f"val_{target}": [] for target in trainer.cfg.target_variables}
    )

    model_api = ModelLogAPI(cfg)

    logger_type = model_api.model_loggers.get("model")

    best_loss = float("inf")

    epoch_iterator = tqdm(range(cfg.epochs)) if tqdm_support else range(cfg.epochs)

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
            if logger_type == "local":
                main_model = model_api.get_model_component_class("model")
                torch.save(best_model_weights, model_api.cfg[main_model]["model_uri"])



        if not tqdm_support:
            print(f"Epoch: {epoch}")
        print(f"Losses - train: {train_loss.item():.6f}  val: {val_loss.item():.6f}")

    model.load_state_dict(best_model_weights)
    model_log_names = model_api.get_model_log_names()
    for module_name, model_class_name in model_log_names.items():
        if module_name == "model": # main model
            model_api.log_model(module_name, model)
        else: # submodule
            model_api.log_model(module_name, model.get_submodule(module_name))

    return model, loss_history, metric_history
