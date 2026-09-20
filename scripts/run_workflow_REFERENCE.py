import subprocess as sp
import xarray as xr
import numpy as np
from numcodecs import Blosc
import os
import sys
import logging
from pathlib import Path
from typing import List, Dict
import shutil
import tomllib  
import toml 
from omegaconf import OmegaConf

ITERATIONS = 10
WD_WFLOW = "/mnt/CEPH_PROJECTS/InterTwin/Wflow/models/emo1"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


DATASET_CONFIG = {
    "dynamics": {"variables": ["precip", "pet", "temp"]},
    "statics": {"variables": ["KsatVer", "c", "f", "RootingDepth", "Sl"]},
    "targets": {"variables": ["vwc"]},
}


SKIP_WFLOW = True
SKIP_SURROGATE_INPUTS = True
SKIP_TRAIN = True
SKIP_CALIBRATION = False


def validate_dataset(dataset: xr.Dataset, expected_vars: List[str], dataset_name: str) -> bool:
    """Validate that the dataset contains the expected variables."""
    missing_vars = [var for var in expected_vars if var not in dataset.variables]
    if missing_vars:
        logger.error(f"{dataset_name.upper()} Error: Missing variables {missing_vars}")
        return False
    logger.info(f"{dataset_name.capitalize()} variables: {list(dataset.variables)}")
    return True

def unpack_soil_layers(dataset, dataset_name, soil_layers: List[int], variable) -> List[int]:
    if len(soil_layers) == 1:
        dataset = dataset.sel(layer=soil_layers).squeeze("layer")
    else:
        layers = range(len(soil_layers))
        for sl in layers:
            dataset[f"{variable}{sl}"] = dataset[variable].isel(layer=sl, drop=True).rename(f"variable{sl + 1}")
        dataset = dataset.drop_vars(variable)
    return dataset

def prepare_dataset(dataset: xr.Dataset, dataset_name, soil_layers = [1]) -> xr.Dataset:
    if dataset_name == "targets":
        # ! Wflow produces netcdf with lat flipped
        dataset.isel(lat=slice(None, None, -1))
    
    if dataset_name == "dynamics" or dataset_name == "statics":
        dataset = dataset.rename_dims({"latitude":"lat", "longitude":"lon"})
        try:
            dataset = dataset.rename_vars({"latitude":"lat", "longitude":"lon"})
        except:
            pass
    # Remove the layer dimension and create new variable for each soil layer
    # TODO: The variable parameter is hardcoded here for statics and dynamics.  
    if dataset_name == "statics":
        logger.info(f"Unpacking and removing soil layer for {dataset_name}")
        dataset = unpack_soil_layers(dataset, dataset_name, soil_layers, variable = "c")
    elif dataset_name == "targets":
        logger.info(f"Unpacking and removing soil layer for {dataset_name}")
        dataset = unpack_soil_layers(dataset, dataset_name, soil_layers, variable = "vwc")
    
    #if dataset_name == "statics":
        #dataset = dataset.drop_dims("time")
   
    # Convert to float32 
    dataset = dataset.apply(lambda x: x.astype(np.float32) if x.dtype == np.float64 else x)
    
    # Remove attributes that is not serializable
    dataset.attrs.pop("_FillValue", None)

    return dataset

def generate_mask(dataset):
    mask_from_static = ["thetaS", "wflow_lakeareas"]
    mask_rename = ["mask_missing", "mask_lake"]
    masks = []

    for i, mask in enumerate(mask_from_static):
        if i == 0:
            masks.append(np.isnan(dataset[mask]).rename(mask_rename[i]))
        else:
            try:
                masks.append((dataset[mask] > 0).astype(np.bool_).rename(mask_rename[i]))
            except:
                # wflow_lakeareas is not in the dataset if domain is too small and there are no lakes
                mask_rename.pop(1)
                mask_from_static.pop(1)
    das = []
    for (
        mask,
        name,
    ) in zip(masks, mask_rename):
        das.append(mask.rename(name))

    return xr.merge(das).to_dataarray(dim="mask_layer", name="mask").to_dataset(dim="mask_layer")

def write_to_zarr(dataset: xr.DataArray, output_path: str,name: str, overwrite="w"):
    """Write the dataset to a Zarr file."""
    dataset.attrs.pop("_FillValue", None)  # Remove problematic attributes

    #fixing issue about overwriting permission error of staticmaps_calib
    dataset = dataset.load()

    filename = f"{output_path}/{name}.zarr"
    dataset.to_zarr(
        store=filename,
        #group=group,
        mode=overwrite,
    )
    logger.info(f"Written {name} to {output_path}")

def process_and_convert(
    datasets: Dict[str, str]
):
    """Process and convert multiple datasets to Zarr."""
    PROCESSED = {}
    
    #PROCESSED["dynamics"] = "/mnt/CEPH_PROJECTS/InterTwin/hydrologic_data/surrogate_input/emo1_dynamic_calib.zarr"

    for dataset_name, dataset_path in datasets.items():
        logger.info(f"Processing {dataset_name} from {dataset_path}")
        if not os.path.isfile(dataset_path):
            logger.error(f"{dataset_name.upper()} Error: Path is not a file {dataset_path}")
            continue
        try:
            with xr.open_dataset(dataset_path) as dataset:
                #dataset.close()
                dataset = prepare_dataset(dataset, dataset_name, soil_layers= [1])
            
            if dataset_name == "targets":
                # flipping lat ....
                dataset = dataset.sel(lat=slice(None, None, -1))
            
            if dataset_name == "statics":

                mask = generate_mask(dataset)
                PROCESSED["mask"] = mask
            config = DATASET_CONFIG[dataset_name]
            if not validate_dataset(dataset, config["variables"], dataset_name):
                continue
            PROCESSED[dataset_name] = dataset
        except Exception as e:
            logger.error(f"{dataset_name.upper()} Error: {e}")

    
    # TODO: Subset dynamics time to targets time 
    #PROCESSED["dynamics"] = PROCESSED["dynamics"].sel(time=PROCESSED["targets"].time)

    #dynamics_merged = xr.merge([PROCESSED["dynamics"], PROCESSED["targets"]])
    statics_merged = xr.merge([PROCESSED["statics"], PROCESSED["mask"]])
    
    return PROCESSED["targets"], statics_merged

if __name__ == "__main__":

    for iter in range(ITERATIONS):

        logger.info(f"ITERATION: {iter}")

        if iter == 0:
            # first iteration take ptf_param 
            shutil.copy(Path(WD_WFLOW) / "staticmaps.nc" , Path(WD_WFLOW) / "staticmaps_calib_0.nc")     

        try:
            if iter > 0:

                if not SKIP_WFLOW:
                    logger.info("RUN WFLOW MODEL")

                    if Path("/mnt/CEPH_PROJECTS/InterTwin/Wflow/models/emo1/run_default/output_calib.nc").exists() and not iter == (ITERATIONS -1):
                        # remove previous calibration run
                        # overwrite output every iteration to avoid memory explosion
                        # keep last run
                        os.remove("/mnt/CEPH_PROJECTS/InterTwin/Wflow/models/emo1/run_default/output_calib.nc")

                    with open("/mnt/CEPH_PROJECTS/InterTwin/Wflow/models/emo1/wflow_sbm_workflow.toml", "rb") as f:
                        data = tomllib.load(f)

                    data["input"]["path_static"] = f"{WD_WFLOW}/staticmaps_calib_{iter}.nc"

                    data["output"]["path"] = f"{WD_WFLOW}/run_default/output_calib_{iter}.nc"

                    data["csv"]["path"] = f"{WD_WFLOW}/run_default/output_calib_{iter}.csv"

                    with open(f"{WD_WFLOW}/wflow_sbm_workflow.toml", "w") as f:
                        toml.dump(data, f)

                    ret = sp.run(["julia",
                                f'--project=/mnt/CEPH_PROJECTS/InterTwin/Wflow/models/emo1', 
                                "-t 24", 
                                '-e using Wflow;Wflow.run("/mnt/CEPH_PROJECTS/InterTwin/Wflow/models/emo1/wflow_sbm_workflow.toml")'],  check=True
                                )
                    
                    if ret.returncode > 0:
                        sys.exit()
                else:
                    logger.info("SKIPPING WFLOW MODEL")

            else:
                logger.info(f"iter: {iter}, skipping running wflow model")
            
            # # == PREPARE OUTPUT
            if not SKIP_SURROGATE_INPUTS:
                logger.info("PREPARE SURROGATE INPUTS")

                datasets = {
                    "statics":f"/mnt/CEPH_PROJECTS/InterTwin/Wflow/models/emo1/staticmaps_calib_{iter}.nc", # iter == 0 orig params
                    "targets":f"/mnt/CEPH_PROJECTS/InterTwin/Wflow/models/emo1/run_default/output_calib_{iter}.nc" # iter == 0 uncalibrated run
                }
                # create zarr inputs for hython model
                target, static = process_and_convert(
                    datasets
                )
                # append new targets to dynamic
                write_to_zarr(target, "/mnt/CEPH_PROJECTS/InterTwin/hydrologic_data/surrogate_input/", name = "emo1_dynamic_calib", overwrite="a")
                # overwrite static
                write_to_zarr(static, "/mnt/CEPH_PROJECTS/InterTwin/hydrologic_data/surrogate_input/", name = "emo1_static_calib")

            if not SKIP_TRAIN:
                # == TRAIN SURROGATE
                logger.info("TRAIN SURROGATE")

                cfg = OmegaConf.load("/home/iferrario/dev/article/param_estimation/config/config_training_calibration_loop.yaml") 
                if iter == 0:
                    # random initialisation of model weights
                    cfg.model_logger.CudaLSTM.load = False

                    cfg.experiment_run =  f"train_multicycle"
                else:
                    # initialise from model trained in iter -1
                    cfg.model_logger.CudaLSTM.load = True

                cfg.data_source.file.dynamic_inputs = "/mnt/CEPH_PROJECTS/InterTwin/hydrologic_data/surrogate_input/emo1_dynamic_calib.zarr"
                cfg.data_source.file.static_inputs = "/mnt/CEPH_PROJECTS/InterTwin/hydrologic_data/surrogate_input/emo1_static_calib.zarr"

                OmegaConf.save(cfg, "/home/iferrario/dev/article/param_estimation/config/config_training_calibration_loop.yaml")

                # # CLEAN STATISTICS FROM RUN FOLDER
                if Path(f"{cfg.work_dir}/{cfg.experiment_name}_{cfg.experiment_run}").exists():
                    # remove yaml files
                    [os.remove(f"{cfg.work_dir}/{cfg.experiment_name}_{cfg.experiment_run}/{file}") for file in os.listdir(f"{cfg.work_dir}/{cfg.experiment_name}_{cfg.experiment_run}") if "yaml" in file]
                
                ret = sp.run([
                    "itwinai exec-pipeline --config-dir /home/iferrario/dev/article/param_estimation/config --config-name config_training_calibration_loop",
                ], shell=True)


                ## COPY MODEL TO KEEP TRACK OF MULTICYCLE TRAINING
                shutil.copy("/mnt/CEPH_PROJECTS/InterTwin/hython_model_run/loop_train_multicycle/CudaLSTM.pt",
                    f"/mnt/CEPH_PROJECTS/InterTwin/hython_model_run/loop_train_multicycle/model_sequence/CudaLSTM_{iter}.pt")

                if ret.returncode > 0:
                    sys.exit()
                
            #== CALIBRATE SURROGATE, RUN INFERENCE, WRITE CALIBRATED PARAMETERS

            # RESCALE RT0
            # logger.info("RESCALE RT0")
            # ds = xr.open_dataset("/mnt/CEPH_PROJECTS/InterTwin/hydrologic_data/SSM-RT0-SIG0-R-CRRL/processed/alps_rt0old_2017-2022.nc")
            # ds = ds.chunk({"time":-1,"lat":50, "lon":50})
            # #wflow_target = xr.open_dataset("/mnt/CEPH_PROJECTS/InterTwin/hydrologic_data/surrogate_input/emo1_dynamic_calib.zarr/", engine="zarr").vwc
            # wflow_static = xr.open_dataset("/mnt/CEPH_PROJECTS/InterTwin/hydrologic_data/surrogate_input/emo1_static_calib.zarr/", engine="zarr")
            # thetaS = wflow_static.thetaS
            # thetaR = wflow_static.thetaR
            # dsmin = ds.min(dim="time")
            # ds_scaled = (ds - dsmin) / (ds.max(dim="time") - dsmin )*(thetaS - thetaR) + thetaR
            # ds_scaled.to_netcdf(f"/mnt/CEPH_PROJECTS/InterTwin/hydrologic_data/SSM-RT0-SIG0-R-CRRL/processed/alps_rt0old_2017-2022_theta_{iter}.nc")
            # del wflow_static, ds, dsmin, ds_scaled

            if not SKIP_CALIBRATION:

                logger.info("CALIBRATE")
                
                
                
                cfg = OmegaConf.load("/home/iferrario/dev/article/param_estimation/config/config_calibration_loop.yaml") 
                cfg.experiment_run =  f"cal_multicycle" #{iter}"
                cfg.data_source.file.target_variables = f"/mnt/CEPH_PROJECTS/InterTwin/hydrologic_data/SSM-RT0-SIG0-R-CRRL/processed/alps_rt0old_2017-2022_theta.nc"#_{iter}.nc"


                if iter == 0:
                    # random initialisation of model weights
                    cfg.model_logger.TransferNN.load = False

                else:
                    # initialise from model trained in iter -1
                    cfg.model_logger.TransferNN.load = True

                OmegaConf.save(cfg, "/home/iferrario/dev/article/param_estimation/config/config_calibration_loop.yaml")

                # CLEAN RUN FOLDER
                if Path(f"{cfg.work_dir}/{cfg.experiment_name}_{cfg.experiment_run}").exists():
                    # remove yaml files
                    [os.remove(f"{cfg.work_dir}/{cfg.experiment_name}_{cfg.experiment_run}/{file}") for file in os.listdir(f"{cfg.work_dir}/{cfg.experiment_name}_{cfg.experiment_run}") if "yaml" in file]

                ret = sp.run(
                    "itwinai exec-pipeline --config-dir /home/iferrario/dev/article/param_estimation/config --config-name config_calibration_loop", shell=True
                )

                ## COPY TO KEEP TRACK OF MULTICYCLE TRAINING
                shutil.copy("/mnt/CEPH_PROJECTS/InterTwin/hython_model_run/loop_cal_multicycle/Hybrid.pt",
                    f"/mnt/CEPH_PROJECTS/InterTwin/hython_model_run/loop_cal_multicycle/model_sequence/Hybrid_{iter}.pt")

                if ret.returncode > 0:
                    sys.exit()
                
            # # == PERTURB PARAMETERS, UPDATE STATICMAPS

            logger.info("PERTURB CALIBRATED PARAMETER")

            factor = 0.001
            random_perc = 0.05
            # load calibrated parameters from calibration inference step
            ds_params = xr.open_dataset("/mnt/CEPH_PROJECTS/InterTwin/Wflow/models/emo1/run_default/inference_parameter.nc")
            
            # shutil.copy("/mnt/CEPH_PROJECTS/InterTwin/Wflow/models/emo1/run_default/inference_parameters.nc" ,
            #              "/mnt/CEPH_PROJECTS/InterTwin/Wflow/models/emo1/run_default/i.nc")
            # # update
            from copy import deepcopy

            w = xr.open_dataset("/mnt/CEPH_PROJECTS/InterTwin/Wflow/models/emo1/staticmaps.nc")

            w.close()

            thetaS = w.thetaS

            orig_mask = thetaS.isnull()

            orig_static = deepcopy(w)

            for ipar in ds_params.data_vars:

                # Interpolate to fill predictor gaps 
                interp = ds_params.rename({"lat":"latitude", "lon":"longitude"})[ipar].interpolate_na(dim="longitude", method="linear")

                #import pdb;pdb.set_trace()
                #interp = interp + np.random.randn(*interp.shape)*factor
                #
                interp = interp + np.random.uniform(0.04, 0.06, size=interp.shape)*np.random.choice([1,-1], size=interp.shape)*interp

                found_negative = bool((interp <= 0).any())

                if found_negative:
                    print(f"{chr(0x26A0)}    Found {int( (interp <= 0).sum())} negative values in {ipar}, filling with average value.")

                mask_neg = interp <= 0
                mask =  ~( orig_mask | mask_neg)

                if ipar == "c":    
                    orig_static[ipar][0] = interp.where(~interp.isnull(), interp.mean()).where(mask)
                else:
                    orig_static[ipar] = interp.where(~interp.isnull(), interp.mean()).where(mask)

                if found_negative:
                    m = interp.mean()
                    print("interpolate with ", m.item(0))
                    orig_static[ipar] = orig_static[ipar].where(~orig_mask, m.item(0)).where(~orig_mask)
                    orig_static[ipar] = orig_static[ipar].interpolate_na(dim="longitude").where(~orig_mask)
                    orig_static[ipar] = orig_static[ipar].where( ~orig_static[ipar].where(~orig_mask, 100).isnull() , 10).where(~orig_mask)
        
            logger.info("UPDATE STATICMAPS")
            
            # Use this in the next iteration, avoid overwriting same netcdf
            path = Path(f"/mnt/CEPH_PROJECTS/InterTwin/Wflow/models/emo1/staticmaps_calib_{iter + 1}.nc")
            if path.exists():
                path.unlink()

            orig_static.to_netcdf(f"/mnt/CEPH_PROJECTS/InterTwin/Wflow/models/emo1/staticmaps_calib_{iter + 1}.nc", mode="w")
            
            path = Path("/mnt/CEPH_PROJECTS/InterTwin/Wflow/models/emo1/run_default/inference_parameter.nc")
            if path.exists():
                path.unlink()

        except KeyboardInterrupt:
            sys.exit(1)
