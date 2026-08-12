import os
import pickle
import random
import copy
from datetime import datetime
import numpy as np
from tqdm.auto import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter

from diffusers.training_utils import EMAModel

from maze_datasets import MazeDataset

from Neural_Network.local_map_encoder import ConditionalUnet1DWithLocalMap, ConditionalUnet1D
from belief_env import Belief_env
from Model.diffusion import Diffusion
from Model.flowmatching import FlowMatching


def init_network(
        input_channels,
        output_channels,
        action_dim,
        obs_dim,
        obs_history,
        action_history=0,
        goal_conditioned=True,
        goal_dim=2,
        local_map_conditioned=True,
        local_map_encoder="identity",
        local_map_embedding_dim=9,
        local_map_size=None,
        **kwargs,
):
    global_cond_dim = obs_dim * obs_history + goal_dim * goal_conditioned + action_history * action_dim
    if local_map_conditioned:
        if local_map_encoder.lower() == "identity" or local_map_encoder.lower() == "mlp":
            embedding_dim = local_map_size ** 2
        else:
            embedding_dim = local_map_embedding_dim

        network = ConditionalUnet1DWithLocalMap(
            input_channels=input_channels,
            output_channels=output_channels,
            encoder_name=local_map_encoder,
            embedding_dim=embedding_dim,
            additional_global_cond_dim=global_cond_dim,
            local_map_size=local_map_size,
            **kwargs
        )
    else:
        network = ConditionalUnet1D(
            input_channels=input_channels,
            output_channels=output_channels,
            global_cond_dim=global_cond_dim,
            **kwargs
        )
    return network


def get_dataset(env_id):
    if "belief" not in env_id.lower():
        raise ValueError(f"Invalid env_id for this training script: {env_id}")

    filename = "datasets/belief_episodes.pkl"
    with open(filename, "rb") as f:
        dataset = pickle.load(f)

    def get_obs(sample):
        return sample['observations']

    def get_act(sample):
        return sample['actions']

    # if metadata/env_id exists load, otherwise create and save
    metadata_path = f"metadata/{env_id}.pt"
    if os.path.exists(metadata_path):
        metadata = torch.load(metadata_path, weights_only=False)

        # for v_x, v_y using v range
        # metadata['Observations_min'][2] = metadata['Observations_min'][3]
        # metadata['Observations_max'][2] = metadata['Observations_max'][3]
    else:
        # Initialize arrays to store the sum and sum of squares
        Observations_sum = np.zeros_like(get_obs(dataset[0])[0])
        Observations_sum_sq = np.zeros_like(get_obs(dataset[0])[0])
        observations_min = np.min(get_obs(dataset[0]), axis=0)
        observations_max = np.max(get_obs(dataset[0]), axis=0)
        Actions_sum = np.zeros_like(get_act(dataset[0])[0])
        Actions_sum_sq = np.zeros_like(get_act(dataset[0])[0])
        actions_min = np.min(get_act(dataset[0]), axis=0)
        actions_max = np.max(get_act(dataset[0]), axis=0)

        # Iterate through each episode in the dataset
        total_steps = 0
        total_actions = 0
        for episode in tqdm(dataset):
            Observations_sum += np.sum(get_obs(episode), axis=0)
            Observations_sum_sq += np.sum(get_obs(episode) ** 2, axis=0)
            observations_min = np.minimum(observations_min, np.min(get_obs(episode), axis=0))
            observations_max = np.maximum(observations_max, np.max(get_obs(episode), axis=0))
            Actions_sum += np.sum(get_act(episode), axis=0)
            Actions_sum_sq += np.sum(get_act(episode) ** 2, axis=0)
            actions_min = np.minimum(actions_min, np.min(get_act(episode), axis=0))
            actions_max = np.maximum(actions_max, np.max(get_act(episode), axis=0))
            total_steps += len(get_obs(episode))
            total_actions += len(get_act(episode))

        # Calculate mean and standard deviation
        observations_mean = Observations_sum / total_steps
        observations_std = np.sqrt(Observations_sum_sq / total_steps - observations_mean ** 2)
        actions_mean = Actions_sum / total_actions
        actions_std = np.sqrt(Actions_sum_sq / total_actions - actions_mean ** 2)

        # Pack into a dictionary
        metadata = {
            "Observations_mean": observations_mean,
            "Observations_std": observations_std,
            "Observations_min": observations_min,
            "Observations_max": observations_max,
            "Actions_mean": actions_mean,
            "Actions_std": actions_std,
            "Actions_min": actions_min,
            "Actions_max": actions_max,
        }
        os.makedirs("metadata", exist_ok=True)
        torch.save(metadata, metadata_path)

    return dataset, metadata


def build_dataloaders(
        env_id,
        seed,
        batch_size,
        num_workers,
        prediction_type,
        obs_history,
        action_history,
        position_conditioned,
        local_map_size,
        local_map_scale,
        pred_horizon,
        augmentations,
):
    if "belief" not in env_id.lower():
        raise ValueError(f"Invalid env_id: {env_id}")

    env = Belief_env()
    global_map = env.build_belief_global_map()
    map_center = (np.mean(env.x_bounds), np.mean(env.y_bounds))

    full_obs_dim = 2
    obs_dim = full_obs_dim if position_conditioned else 0
    action_dim = 2

    dataset, metadata = get_dataset(env_id)

    train_dataset = MazeDataset(
        dataset,
        metadata,
        env_id,
        obs_history=obs_history,
        action_history=action_history,
        pred_horizon=pred_horizon,
        prediction_type=prediction_type,
        position_conditioned=position_conditioned,
        local_map_size=local_map_size,
        scale=local_map_scale,
        global_map=global_map,
        s_global=1.0,
        map_center=map_center,
        augmentations=augmentations,
    )

    val_size = max(1, int(0.05 * len(train_dataset)))
    train_size = len(train_dataset) - val_size
    split_generator = torch.Generator().manual_seed(seed)
    train_split, val_split = torch.utils.data.random_split(
        train_dataset,
        [train_size, val_size],
        generator=split_generator,
    )

    persistent_workers = num_workers > 0
    train_dataloader = torch.utils.data.DataLoader(
        train_split,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_split,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )

    return train_dataloader, val_dataloader, full_obs_dim, obs_dim, action_dim


def load_from_checkpoint(checkpoint, output_dir, model, ema, optimizer, device):
    if checkpoint is None:
        return 1, 0

    checkpoint_name = checkpoint
    checkpoint_path = checkpoint
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(output_dir, checkpoint_path)

    checkpoint_state = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint_state.get(
        'model_state_dict',
        checkpoint_state.get('noise_pred_net_state_dict'),
    )
    if state_dict is None:
        raise KeyError("Checkpoint is missing model_state_dict/noise_pred_net_state_dict")

    model.model.load_state_dict(state_dict)
    ema.load_state_dict(checkpoint_state['ema_state_dict'])
    optimizer.load_state_dict(checkpoint_state['optimizer_state_dict'])

    start_epoch = checkpoint_state['epoch'] + 1
    start_step = checkpoint_state.get('step', 0)
    if start_step == 0 and '_step_' in checkpoint_name:
        step_text = checkpoint_name.rsplit('_step_', 1)[1]
        step_digits = ""
        for char in step_text:
            if not char.isdigit():
                break
            step_digits += char
        if step_digits:
            start_step = int(step_digits)

    loss = checkpoint_state['loss']
    print(f"Loaded checkpoint from epoch {checkpoint_state['epoch']} with loss {loss}")
    return start_epoch, start_step


def save_checkpoint(output_dir, filename, epoch, step, loss, model, ema, optimizer):
    checkpoint_state = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.model.state_dict(),
        'ema_state_dict': ema.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    path = os.path.join(output_dir, filename)
    torch.save(checkpoint_state, path)
    return path


def train(
        debug=False,
        # Resume training
        checkpoint=None,
        
        # Settings
        seed=42,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        output_dir='checkpoints/',
        experiment_name=f"diffusion_planning_{datetime.now().strftime('%d_%m_%H_%M')}",
        env_id="beliefmaze",
        prediction_type='observations',
        obs_history=1,  # number of observations to use per sample
        action_history=1,  # number of past actions to use per sample
        position_conditioned=False,
        goal_conditioned=True,
        goal_dim=2,
        local_map_conditioned=True,
        local_map_size=10,
        local_map_scale=0.2,
        local_map_embedding_dim=64,
        local_map_encoder="identity",
        augmentations=None,

        # Training settings
        num_epochs=10,
        batch_size=128, #256,
        num_workers=3,
        checkpoint_interval_epochs=3,

        # Diffusion settings
        policy="diffusion",  # "diffusion"/"flow_matching"
        num_diffusion_iters=100,
        unet_base_channels=256,
        unet_down_dims=[1, 2, 4],

        # MPC settings
        pred_horizon=16,
):
    # set seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
  
    os.makedirs(output_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=f"runs/{experiment_name}")
    
    train_dataloader, val_dataloader, full_obs_dim, obs_dim, action_dim = build_dataloaders(
        env_id=env_id,
        seed=seed,
        batch_size=batch_size,
        num_workers=num_workers,
        prediction_type=prediction_type,
        obs_history=obs_history,
        action_history=action_history,
        position_conditioned=position_conditioned,
        local_map_size=local_map_size,
        local_map_scale=local_map_scale,
        pred_horizon=pred_horizon,
        augmentations=augmentations,
    )

    network = init_network(
        input_channels=action_dim if prediction_type == "actions" else full_obs_dim,
        output_channels=action_dim if prediction_type == "actions" else full_obs_dim,
        action_dim=action_dim,
        obs_dim=obs_dim,
        obs_history=obs_history,
        action_history=action_history,
        goal_conditioned=goal_conditioned,
        goal_dim=goal_dim,
        local_map_conditioned=local_map_conditioned,
        local_map_encoder=local_map_encoder,
        local_map_embedding_dim=local_map_embedding_dim,
        local_map_size=local_map_size,
        base_channels=unet_base_channels,
        down_dims=unet_down_dims,
    )

    model_kwargs = {
        "env_id": env_id,
        "pred_horizon": pred_horizon,
        "action_dim": action_dim,
        "prediction_type": prediction_type,
        "obs_history": obs_history,
        "action_history": action_history,
        "position_conditioned": position_conditioned,
        "goal_conditioned": goal_conditioned,
        "local_map_conditioned": local_map_conditioned,
        "local_map_size": local_map_size,
        "device": device,
    }
    if policy == "diffusion":
        model = Diffusion(network, num_diffusion_iters=num_diffusion_iters, **model_kwargs)
    elif policy == "flow_matching":
        model = FlowMatching(network, ode_steps=num_diffusion_iters, **model_kwargs)
    else:
        raise ValueError(f"Unknown policy: {policy}")
    
    model.to(device).train()

    ema = EMAModel(
        parameters=model.parameters(),
        power=0.75)
    ema_model = copy.deepcopy(model).to(device).eval()

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=3.0e-5, betas=(0.95, 0.999), eps=1.0e-8, weight_decay=1e-6)  #   #lr=1e-4, weight_decay=1e-6

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=len(train_dataloader) * num_epochs,
        eta_min=1e-6,
    )

    start_epoch, start_step = load_from_checkpoint(
        checkpoint=checkpoint,
        output_dir=output_dir,
        model=model,
        ema=ema,
        optimizer=optimizer,
        device=device,
    )

    step = start_step
    best_val_loss = float('inf')
    for epoch in range(start_epoch, num_epochs + 1):  # [1,num_epochs]
        epoch_loss = list()
        with tqdm(train_dataloader, desc=f'Epoch {epoch}') as tepoch:
            for nobs, naction, goal, local_map in tepoch:
                step += 1
                loss = model.loss(nobs, naction, goal, local_map)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

                # update Exponential Moving Average of the model weights
                ema.step(model.parameters())

                # logging
                loss_cpu = loss.item()
                epoch_loss.append(loss_cpu)
                tepoch.set_postfix(loss=loss_cpu)

        ema_model.load_state_dict(model.state_dict())
        ema.copy_to(ema_model.parameters())

        #Eval set
        ema_model.eval()
        val_losses = []
        with torch.no_grad():
            for nobs, naction, goal, local_map in val_dataloader:
                val_losses.append(ema_model.loss(nobs, naction, goal, local_map).item())
        model.train()
        val_loss = float(np.mean(val_losses))
        
        if writer is not None:
            writer.add_scalar('Loss/train', np.mean(epoch_loss), epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = save_checkpoint(
                output_dir=output_dir,
                filename="best.pt",
                epoch=epoch,
                step=step,
                loss=val_loss,
                model=ema_model,
                ema=ema,
                optimizer=optimizer,
            )
            print(f"Saved best checkpoint: {best_checkpoint_path} (val_loss={val_loss:.6f})")

        if epoch % checkpoint_interval_epochs == 0 or epoch == num_epochs:
            latest_checkpoint_path = save_checkpoint(
                output_dir=output_dir,
                filename="latest.pt",
                epoch=epoch,
                step=step,
                loss=val_loss,
                model=ema_model,
                ema=ema,
                optimizer=optimizer,
            )
            print(f"Saved latest checkpoint: {latest_checkpoint_path}")

    # Weights of the EMA model
    # is used for inference
    ema_model.load_state_dict(model.state_dict())
    ema.copy_to(ema_model.parameters())
    if writer is not None:
        writer.close()

    print("Done")
