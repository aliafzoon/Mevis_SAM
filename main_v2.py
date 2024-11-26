import argparse
from pathlib import Path

import mlflow
import torch
from monai.losses import DiceCELoss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import cfg
from dataset_mevis_v2 import MRI_dataset_batched
from models.sam import build_sam_mevis
from utils import eval_seg

# TODO:

"""
    This scripts is the starting point on the training loop. 
    It handles MLflow arguments, loads model and the dataset,
    writes logs and saves models at intervals.
"""

args = cfg.parse_args()
tracking_server = "http://127.0.0.1:5000"
TRAIN_DATA_FILE = "data_files/Train_data_files_resampled_v2.json"
VALID_DATA_FILE = "data_files/Validation_data_files_resampled_v2.json"
DEVICE = torch.device("cuda:" + str(args.gpu_device))
CHECKPOINT_DIRECTORY = Path("/data/sab_data/checkpoints")
LOGDIR = Path("/data/sab_data/model_logs/mlflow")
# train_folder = "decoder_only/30p_prompt_v2"
BONE_CHECKPOINT = CHECKPOINT_DIRECTORY / "bone_sam.pth"
VALID_EVERY = 2
SAVE_EVERY = 3
OPEN_LAYERS = (
    "mask_decoder.transformer.layers.0.MLP_Adapter,"
    "mask_decoder.transformer.layers.0.Adapter,"
    "mask_decoder.transformer.layers.1.MLP_Adapter,"
    "mask_decoder.transformer.layers.1.Adapter,"
)


def mevis_args_parser():
    """
    Returns arguments for Mevis SAM fine-tuning process.

    Returns:
        argparse.Namespace: A namespace containing all the parsed arguments and their values.
    """
    parser = argparse.ArgumentParser(
        description="Mevis SAM fine-tuning parameters.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="",
        help="Checkpoint to continue the training",
    )
    parser.add_argument(
        "--prompt_probability",
        type=float,
        default=0.3,
        help="probability of generating prompts for each batch",
    )
    parser.add_argument(
        "--lr_schedule",
        type=bool,
        default=True,
        help="Use earning rate scheduler during training.",
    )
    parser.add_argument(
        "--lr_train_start",
        type=float,
        default=5e-4,
        help="Learning rate starting value during training.",
    )
    parser.add_argument(
        "--lr_train_end",
        type=float,
        default=5e-5,
        help="Learning rate ending value during training.",
    )
    parser.add_argument(
        "--lr_warmup", type=float, default=1e-5, help="Learning rate on warmup."
    )
    parser.add_argument(
        "--epochs", type=int, default=150, help="Number of training epochs"
    )
    parser.add_argument(
        "--epochs_warmup", type=int, default=20, help="Number of warmup epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=120, help="Number of slices in a batch"
    )
    mevis_args = parser.parse_args("")

    return mevis_args


def lr_scheduler(
    optimizer,
    epoch,
    n_warmup_epochs=20,
    n_epochs=60,
    base_lr=1e-4,
    warmup_lr_start=1e-5,
    end_lr=1e-5,
):
    """
    Adjusts the learning rate between epochs.The scheduler has two phases:
    1. Warmup Phase: A linear increase in learning rate from `warmup_lr_start` to `base_lr`
       over the first `n_warmup_epochs` epochs.
    2. Decay Phase: A linear decrease in learning rate from `base_lr` to `end_lr`
       over the remaining epochs (total epochs = `n_warmup_epochs` + `n_epochs`).

    Args:
        optimizer (torch.optim.Optimizer): The optimizer.
        epoch (int): The current training epoch (0-indexed).
        n_warmup_epochs (int): Number of warmup epochs (default: 20).
        n_epochs (int): Number of training epochs after the warmup phase (default: 60).
        base_lr (float): The base learning rate (default: 1e-4).
        warmup_lr_start (float): The starting learning rate during the warmup phase (default: 1e-5).
        end_lr (float): The final learning rate at the end of training (default: 1e-5).

    Returns:
        torch.optim.Optimizer: The optimizer with the updated learning rate.
    """
    if epoch <= n_warmup_epochs:
        # Linear warmup phase
        lr = warmup_lr_start + (base_lr - warmup_lr_start) * \
            (epoch / n_warmup_epochs)
    else:
        # Linear decay phase
        decay_epochs = n_epochs + n_warmup_epochs
        lr = base_lr - (base_lr - end_lr) * (epoch / decay_epochs)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return optimizer


def write_hists(net, writer, epoch):
    """
    Logs histograms of specific network layer weights for TensorBoard.
    Should adjust for different setup

    Args:
        net (torch.nn.Module): The model.
        writer (torch.utils.tensorboard.SummaryWriter): TensorBoard writer.
        epoch (int): The current epoch, used to index the histograms.
    """
    writer.add_histogram(
        "mask_decoder.transformer.layers[1].MLP_Adapter/D_fc1/weight",
        net.mask_decoder.transformer.layers[1].MLP_Adapter.D_fc1.weight,
        epoch,
    )
    writer.add_histogram(
        "mask_decoder.transformer.layers[1].MLP_Adapter/D_fc1/bias",
        net.mask_decoder.transformer.layers[1].MLP_Adapter.D_fc1.bias,
        epoch,
    )

    writer.add_histogram(
        "mask_decoder.transformer.layers[1].MLP_Adapter/D_fc2/weight",
        net.mask_decoder.transformer.layers[1].MLP_Adapter.D_fc2.weight,
        epoch,
    )
    writer.add_histogram(
        "mask_decoder.transformer.layers[1].MLP_Adapter/D_fc2/bias",
        net.mask_decoder.transformer.layers[1].MLP_Adapter.D_fc2.bias,
        epoch,
    )

    writer.add_histogram(
        "mask_decoder.transformer.layers[1].Adapter/D_fc1/weight",
        net.mask_decoder.transformer.layers[1].Adapter.D_fc1.weight,
        epoch,
    )
    writer.add_histogram(
        "mask_decoder.transformer.layers[1].Adapter/D_fc1/bias",
        net.mask_decoder.transformer.layers[1].Adapter.D_fc1.bias,
        epoch,
    )

    writer.add_histogram(
        "mask_decoder.transformer.layers[1].Adapter/D_fc2/weight",
        net.mask_decoder.transformer.layers[1].Adapter.D_fc2.weight,
        epoch,
    )
    writer.add_histogram(
        "mask_decoder.transformer.layers[1].Adapter/D_fc2/bias",
        net.mask_decoder.transformer.layers[1].Adapter.D_fc2.bias,
        epoch,
    )
    return


def train_mevis(
    net: torch.nn.Module,
    optimizer,
    loss_func,
    dataset,
    epoch,
    writer,
):
    """
    Trains the Mevis model for one epoch.

    Args:
        net (torch.nn.Module): The model.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        loss_func (callable): Loss function.
        dataset (MRI_dataset_batched): Each item is a dictionary.
        epoch (int): The current epoch number.
        writer (torch.utils.tensorboard.SummaryWriter): TensorBoard writer for logging metrics.

    Returns:
        torch.Tensor: The loss of the last batch in the epoch.
    """
    epoch_loss = 0
    # train mode
    net.train()
    optimizer.zero_grad()

    with tqdm(total=len(dataset), desc=f"Epoch {epoch}", unit="batch") as pbar:
        for i in range(len(dataset)):
            pack = dataset[i]
            masks = pack["masks"]

            """Train"""
            pred = net.forward(pack, multimask_output=True, if_attention=True)
            loss = loss_func(pred[:, 1:2, :, :], masks)
            pbar.set_postfix(**{"loss (batch)": loss.item()})
            writer.add_scalar(
                "Batch Loss/Training", loss.item(), epoch * len(dataset) + i
            )
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            pbar.update()

    return loss


def validate_mevis(
    net: torch.nn.Module,
    loss_func,
    dataset,
    epoch,
    writer,
):
    """
    Validates the Mevis model on the given dataset.

    Args:
        net (torch.nn.Module): The  model.
        loss_func (callable): Loss function.
        dataset (MRI_dataset_batched): Each item is a dictionary.
        epoch (int): The current epoch number.
        writer (torch.utils.tensorboard.SummaryWriter): TensorBoard writer for logging metrics.

    Returns:
        float: The average loss across the validation dataset.
        dict: A dictionary containing metrics (IoU, Dice) for different thresholds.
    """
    net.eval()
    sum_loss = 0
    thresholds = (0.0, 0.25, 0.5, 0.7, 0.9)
    result_dict = {str(key): {"iou_class": 0, "dice_class": 0}
                   for key in thresholds}
    n_val = len(dataset)

    with tqdm(total=n_val, desc="Validation round", unit="batch", leave=True) as pbar:
        for i in range(n_val):
            pack = dataset[i]
            masks = pack["masks"]
            """ Test """
            with torch.no_grad():
                pred = net.forward(
                    pack, multimask_output=True, if_attention=True)
                loss = loss_func(pred[:, 1:2, :, :], masks)
                pbar.set_postfix(**{"loss (batch)": loss.item()})
                writer.add_scalar(
                    "Batch Loss/validation", loss.item(), epoch * len(dataset) + i
                )
                sum_loss += loss
                temp = eval_seg(pred[:, 1:, :, :], masks, thresholds)
                for th in result_dict.keys():
                    for met, val in result_dict[th].items():
                        result_dict[th][met] = val + temp[th].get(met, 0)

            pbar.update()

    for th in result_dict.keys():
        result_dict[th] = {k: val / n_val for k,
                           val in result_dict[th].items()}

    return sum_loss / n_val, result_dict


def train_procedure(
    run,
    epoch,
    log_base,
    mevis_args,
    merged_args,
):
    """
    Executes the training procedure for the Mevis SAM model.

    Args:
        run (mlflow.entities.Run): Current MLflow run.
        epoch (int): Starting epoch number for training.
        log_base (Path or str): Base directory for saving logs.
        mevis_args (argparse.Namespace): Arguments specific to the training configuration.
        merged_args (argparse.Namespace): General training arguments including dataset and model settings.
    """

    run_id = run.info.run_id
    # Tensorboard logger
    log_save = log_base / run_id
    log_save.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=log_save)

    # Log the hyperparameters
    mlflow.log_params(vars(mevis_args))

    # data
    dataset_validation = MRI_dataset_batched(
        merged_args,
        data_file=VALID_DATA_FILE,
        batch_size=mevis_args.batch_size,
        phase="test",
        operation_mode="queue",
        mask_out_size=merged_args.out_size,
        attention_size=64,
        crop=False,
        crop_size=1024,
        cls=1,
        if_prompt=True,
        prompt_type="points",
        if_attention_map=True,
        device=DEVICE,
    )
    dataset_train = MRI_dataset_batched(
        merged_args,
        data_file=TRAIN_DATA_FILE,
        batch_size=mevis_args.batch_size,
        phase="train",
        operation_mode="queue",
        mask_out_size=merged_args.out_size,
        attention_size=64,
        crop=False,
        crop_size=1024,
        cls=1,
        if_prompt=True,
        prompt_type="points",
        if_attention_map=True,
        device=DEVICE,
    )

    # model
    sam_mevis = build_sam_mevis(
        merged_args,
        mevis_checkpoint=CHECKPOINT_DIRECTORY / BONE_CHECKPOINT,
        num_classes=2,
        device=DEVICE,
    )
    open_layers = [x for x in OPEN_LAYERS.split(",") if x != ""]
    for param in sam_mevis.parameters():
        param.requires_grad = False
    for name, mod in sam_mevis.named_parameters():
        for l in open_layers:
            if l in name:
                mod.requires_grad = True

    total_trainable_params = sum(
        p.numel() for p in sam_mevis.parameters() if p.requires_grad
    )
    mlflow.log_param("total_trainable_params", total_trainable_params)
    print("Number of trainable parameters in the model:", total_trainable_params)

    lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean")
    optimizer = torch.optim.AdamW(
        sam_mevis.parameters(), lr=mevis_args.lr_train_start)

    while epoch < (merged_args.lr_warmup + merged_args.epochs):
        optimizer = lr_scheduler(
            optimizer,
            epoch,
            n_warmup_epochs=merged_args.epochs_warmup,
            n_epochs=merged_args.epochs,
            base_lr=merged_args.lr_train_start,
            warmup_lr_start=merged_args.lr_warmup,
            end_lr=merged_args.lr_train_end,
        )
        dataset_train.on_epoch_end()
        loss = train_mevis(
            net=sam_mevis,
            optimizer=optimizer,
            loss_func=lossfunc,
            dataset=dataset_train,
            epoch=epoch,
            writer=writer,
        )
        mlflow.log_metric(key="DiceCE", value=loss, step=epoch)
        writer.add_scalars("DiceCEloss", {"training": loss}, epoch)
        write_hists(sam_mevis, writer=writer, epoch=epoch)
        if epoch % VALID_EVERY == 0:
            val_loss, val_result = validate_mevis(
                net=sam_mevis,
                loss_func=lossfunc,
                dataset=dataset_validation,
                epoch=epoch,
                writer=writer,
            )
            writer.add_scalars("DiceCEloss", {"validation": val_loss}, epoch)
            for key, value in val_result.items():
                for key_in, value_in in value.items():
                    writer.add_scalar(
                        f"Validation Metrics/{key_in}/thr{key}", value_in, epoch
                    )
        if epoch % SAVE_EVERY == 0:
            mlflow.pytorch.log_model(
                sam_mevis, artifact_path=f"mevis_sam-epoch{epoch}-{run_id}"
            )

        epoch += 1

    mlflow.pytorch.log_model(
        sam_mevis, artifact_path=f"mevis_sam-epoch{epoch}-{run_id}"
    )
    writer.close()
    return


if __name__ == "__main__":

    mevis_args = mevis_args_parser()
    merged_args = argparse.Namespace(**vars(args), **vars(mevis_args))
    if mevis_args.checkpoint_path != "":
        new_path = Path(mevis_args.checkpoint_path)
        if new_path.is_file() and new_path.suffix == ".pth":
            BONE_CHECKPOINT = new_path

    if "bone_sam" in BONE_CHECKPOINT.name:
        epoch = 0
        run_id = None
    else:
        epoch = int(BONE_CHECKPOINT.name.split("epoch")[1].split("-")[0])
        run_id = BONE_CHECKPOINT.name.split("-")[-1]
    print(f"Starting from epoch {epoch} from checkpoint {
          BONE_CHECKPOINT.name}.")

    # MLflow values
    experiment_name = "with_v2"
    experiment_description = (
        "This is fine-tuning of SAB model for vertebrae. "
        "The improvements from first version are applied. "
        "Including rotation for attention maps and better masking for sacrum."
    )
    experiment_tags = {
        "project_name": "mevis_vertebrae",
        "version": "2.0",
        "project_time": "October-2024",
        "opened_layers": OPEN_LAYERS,
        "mlflow.note.content": experiment_description,
    }
    mlflow.set_tracking_uri(uri=tracking_server)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Creating new experiment {experiment_name}.")
        mlflow.create_experiment(
            name=experiment_name,
            tags=experiment_tags,
        )
        experiment = mlflow.set_experiment(experiment_name)
    else:
        print(f"Resuming from experiment {experiment.name}.")

    # Run
    log_base = LOGDIR / experiment.experiment_id
    if run_id is not None:
        print(f"Resuming from run {run_id}.")
        with mlflow.start_run(run_id=run_id) as parent:
            train_procedure(
                parent,
                epoch,
                log_base=log_base,
                mevis_args=mevis_args,
                merged_args=merged_args,
            )
    else:
        print("Starting new run.")
        with mlflow.start_run() as parent:
            train_procedure(
                parent,
                epoch,
                log_base=log_base,
                mevis_args=mevis_args,
                merged_args=merged_args,
            )
