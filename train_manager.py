from datetime import datetime

import yaml

from train import train



if __name__ == "__main__":

    env_id = "beliefmaze"
    unet_base_channels = {
        'small': 64,
        'medium': 256,
        'large': 512,
        'xlarge': 1024,
    }

    with open(f"cfgs/{env_id}.yaml", "r") as file:
        loaded_config = yaml.safe_load(file)

    timestamp = datetime.now().strftime('%d_%m_%H_%M')
    experiment_name = f"beliefmaze_{timestamp}"  # _{timestamp}

    train(
        checkpoint=loaded_config.get('resume_checkpoint', None),
        experiment_name=experiment_name,
        debug=loaded_config.get('debug', False),
        prediction_type=loaded_config.get('prediction_type', "actions"),    #  ["actions","observations"]
        obs_history=loaded_config.get('obs_history', 1),
        action_history=loaded_config.get('action_history', 1),
        position_conditioned=loaded_config.get('position_conditioned', True),
        goal_conditioned=loaded_config.get('goal_conditioned', True),
        local_map_conditioned=loaded_config.get('local_map_conditioned', True),
        local_map_size=loaded_config.get('local_map_size', 20),
        local_map_scale=loaded_config.get('local_map_scale', 0.2),
        local_map_embedding_dim=loaded_config.get('local_map_embedding_dim', 400),
        local_map_encoder=loaded_config.get('local_map_encoder', "resnet"), #["grid", "max", "identity", "mlp","resnet"]
        num_epochs=loaded_config.get('num_epochs', 10),
        checkpoint_interval_epochs=loaded_config.get('checkpoint_interval_epochs', 3),
        env_id=loaded_config.get('env_id', "beliefmaze"),
        augmentations=loaded_config.get('augmentations', ["mirror"]),   # ["rotate", "mirror"]
        policy=loaded_config.get('policy', "flow_matching"),    # ["diffusion","flow_matching"]
        num_diffusion_iters=loaded_config.get('num_diffusion_iters', 5),
        unet_base_channels=unet_base_channels[loaded_config.get('unet_size', 'large')],
        unet_down_dims=loaded_config.get('unet_down_dims', [1, 2, 4]),
        pred_horizon=loaded_config.get('pred_horizon', 64),
    )
