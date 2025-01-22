import torch
import xarray as xr
import numpy as np

from .utils import get_temporal_steps

def conformal_quantile_regression(model, 
                                  model_paths, 
                                  cal_dataset, 
                                  test_dataset, 
                                  batch_size, 
                                  scaler = None,
                                  coverages = [0.8],
                                  steps=-1,
                                  target = "y_hat",
                                  device="cpu"):
    
    # Define coverages and calculate tau values
    quantiles = [(1 - c) / 2 for c in coverages]  # Lower quantiles
    quantile_pairs = [(round(q,3), round(1 - q,3)) for q in quantiles] 

    selection = get_temporal_steps(steps)


    for coverage, (tau_lower, tau_upper) in zip(coverages, quantile_pairs):

        # ===== UPPER QUANTILE ======== 
        model.load_state_dict(torch.load(model_paths["q90"], weights_only=True))

        # CALIBRATION
        n, t, _ = cal_dataset.xd.shape

        y_val_upper_preds = []
        y_val_true = []

        for i in range(0, n, batch_size):
            d = torch.from_numpy(cal_dataset.xd[i : (i + batch_size)].values).float().to(device)
            s = torch.from_numpy(cal_dataset.xs[i : (i + batch_size)].values).float().to(device)
            t = torch.from_numpy(cal_dataset.y[i : (i + batch_size)].values).float().to(device)

            static_bt = s.unsqueeze(1).repeat(1, d.size(1), 1).to(device)

            x_concat = torch.cat(
                (d, static_bt),
                dim=-1,
            )

            out = model(x_concat)[target].detach().cpu().numpy()

            if scaler is not None:
                out = scaler.transform_inverse(out, "target_variables")

            out = out[:, selection]

            y_val_upper_preds.append(out)

            if scaler is not None:
                t = scaler.transform_inverse(t.detach().cpu().numpy(), "target_variables")
            
            t = t[:, selection]
            y_val_true.append(t)

        y_val_true = np.vstack(y_val_true)
        y_val_upper_preds = np.vstack(y_val_upper_preds)


        # TEST
        n, t, _ = test_dataset.xd.shape

        y_te_upper_preds = []
        y_te_true = []

        for i in range(0, n, batch_size):
            d = torch.from_numpy(test_dataset.xd[i : (i + batch_size)].values).float().to(device)
            s = torch.from_numpy(test_dataset.xs[i : (i + batch_size)].values).float().to(device)
            t = torch.from_numpy(test_dataset.y[i : (i + batch_size)].values).float().to(device)

            static_bt = s.unsqueeze(1).repeat(1, d.size(1), 1).to(device)

            x_concat = torch.cat(
                (d, static_bt),
                dim=-1,
            )

            out = model(x_concat)[target].detach().cpu().numpy()

            if scaler is not None:
                out = scaler.transform_inverse(out, "target_variables")

            out = out[:, selection]

            y_te_upper_preds.append(out)

            if scaler is not None:
                t = scaler.transform_inverse(t.detach().cpu().numpy(), "target_variables")
            
            t = t[:, selection]
            y_te_true.append(t)

        y_te_true = np.vstack(y_te_true)
        y_te_upper_preds = np.vstack(y_te_upper_preds)



        # ==== LOWER QUANTILE ==========

        model.load_state_dict(torch.load(model_paths["q10"], weights_only=True))

        # CALIBRATION
        n, t, _ = cal_dataset.xd.shape

        y_val_lower_preds = []

        for i in range(0, n, batch_size):
            d = torch.from_numpy(cal_dataset.xd[i : (i + batch_size)].values).float().to(device)
            s = torch.from_numpy(cal_dataset.xs[i : (i + batch_size)].values).float().to(device)
            t = torch.from_numpy(cal_dataset.y[i : (i + batch_size)].values).float().to(device)

            static_bt = s.unsqueeze(1).repeat(1, d.size(1), 1).to(device)

            x_concat = torch.cat(
                (d, static_bt),
                dim=-1,
            )

            out = model(x_concat)[target].detach().cpu().numpy()

            if scaler is not None:
                out = scaler.transform_inverse(out, "target_variables")

            out = out[:, selection]

            y_val_lower_preds.append(out)

        y_val_lower_preds = np.vstack(y_val_lower_preds)

        # TEST
        n, t, _ = test_dataset.xd.shape

        y_te_lower_preds = []

        for i in range(0, n, batch_size):
            d = torch.from_numpy(test_dataset.xd[i : (i + batch_size)].values).float().to(device)
            s = torch.from_numpy(test_dataset.xs[i : (i + batch_size)].values).float().to(device)
            t = torch.from_numpy(test_dataset.y[i : (i + batch_size)].values).float().to(device)

            static_bt = s.unsqueeze(1).repeat(1, d.size(1), 1).to(device)

            x_concat = torch.cat(
                (d, static_bt),
                dim=-1,
            )

            out = model(x_concat)[target].detach().cpu().numpy()

            if scaler is not None:
                out = scaler.transform_inverse(out, "target_variables")

            out = out[:, selection]

            y_te_lower_preds.append(out)


        y_te_lower_preds = np.vstack(y_te_lower_preds)

        # === Conformal scores

        mask_val = ~np.isnan(y_val_true)
        mask_te = ~np.isnan(y_te_true)

        #import pdb;pdb.set_trace()

        scores = np.maximum(y_val_true[mask_val] - y_val_upper_preds[mask_val] , 
                            y_val_lower_preds[mask_val] - y_val_true[mask_val])
        q = np.quantile(scores, coverage)

        te_prediction_SET = [y_te_lower_preds - q, y_te_upper_preds +q]
        
        coverage2 = (y_te_true >= te_prediction_SET[0]) & (y_te_true <= te_prediction_SET[1])

        # Calculate the percentage of values covered
        empirical_coverage_percentage = np.mean(coverage2) * 100
        print(f"Empirical Coverage: {empirical_coverage_percentage:.2f}%")
        print("----"*5)

    median_predictions = (te_prediction_SET[0] + te_prediction_SET[1])/2

    return empirical_coverage_percentage, median_predictions


def monte_carlo_dropout(model, dataset, batch_size, n_samples, steps=-1, target="y_hat", device="cpu"):

    model.to(device)

    selection = get_temporal_steps(steps)

    # loop through loader
    test_mean_output_mcd = []
    test_std_output_mcd = []

    n, t, _ = dataset.xd.shape

    model.train()

    for i in range(0, n, batch_size):
        d = torch.from_numpy(dataset.xd[i : (i + batch_size)].values).float().to(device)
        s = torch.from_numpy(dataset.xs[i : (i + batch_size)].values).float().to(device)

        static_bt = s.unsqueeze(1).repeat(1, d.size(1), 1).to(device)

        x_concat = torch.cat(
            (d, static_bt),
            dim=-1,
        )

        # mcd
        output_mcd = []
        for _ in range(n_samples):
            # avoid loading the GPU
            output_mcd.append(model(x_concat)[target].detach().cpu())
        output_mcd = torch.stack(output_mcd) # (mcd samples, N, T, C)
        
        # compute mean and standard deviation
        mean_mcd = output_mcd.mean(dim=0)  # ( N, T, C)
        std_mcd = output_mcd.std(dim=0)
        
        # get steps
        mean_mcd = mean_mcd[:, selection]
        test_mean_output_mcd.append(mean_mcd.detach().cpu().numpy())

        std_mcd = std_mcd[:, selection]
        test_std_output_mcd.append(std_mcd.detach().cpu().numpy())
    
    return np.vstack(test_mean_output_mcd), np.vstack(test_std_output_mcd)
