
if __name__ == "__main__":
    import os
    import argparse
    from pathlib import Path
    import yaml  # type: ignore

    parser = argparse.ArgumentParser(description="Run RL training with optional YAML config")
    parser.add_argument("-c", "--config", type=str, default=None, help="Path to YAML config file")
    _args = parser.parse_args()

    def _load_config_dict(cli_path: str | None = None):
        project_root = Path(__file__).resolve().parent.parent
        config_path_env = os.getenv("CONFIG_PATH")
        if cli_path:
            config_path = Path(cli_path)
        elif config_path_env:
            config_path = Path(config_path_env)
        else:
            config_path = project_root / "config.yaml"
        if yaml is not None and config_path.is_file():
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        raise ValueError(f"Config file {config_path} not found")

    _config = _load_config_dict(_args.config)

    # Hierarchical required config
    env_cfg = _config["env"]
    train_cfg = _config["train"]
    ppo_cfg = _config["ppo"]
    logging_cfg = _config["logging"]

    env_id = env_cfg["id"]
    symmetry_cfg = env_cfg["symmetry"]
    symmetric_method = symmetry_cfg["method"]
    symmetric_params = symmetry_cfg.get("params")
    flip_params = symmetry_cfg.get("flip_params")

    from rl_agent.ppo_symmetric import PPOTorchRLSymmetric
    # Ensure custom envs (e.g., HumanoidWalking-v0) are registered before gym.make
    import environment.custom_envs  # noqa: F401
    from environment.env_wrapper_base import EnvWrapperBase

    # Training horizon (required)
    total_timesteps = int(train_cfg["total_timesteps"])

    # Choose environment class based on symmetry method
    method = (symmetric_method or "none").lower()
    if method == "none":
        from environment.env_wrapper_base import EnvWrapperBase as _Env
        env_class = _Env
        env_kwargs = {
            "env_id": env_id,
            "render_mode": env_cfg.get("render_mode"),
            "make_kwargs": env_cfg["make_kwargs"],
        }
    else:
        from environment.env_random_flip import EnvRandomFlip as _Env
        env_class = _Env
        if flip_params is None:
            raise ValueError("env.symmetry.flip_params must be provided or set method to 'none'")
        if symmetric_params is None:
            raise ValueError("env.symmetry.params must be provided or set method to 'none'")
        env_kwargs = {
            "env_id": env_id,
            "render_mode": env_cfg.get("render_mode"),
            "flip_params": flip_params,
            "make_kwargs": env_cfg["make_kwargs"],
        }
    print(f"=== {env_class.__name__} Environment Training ===")
    
    
    from datetime import datetime
    """Train PPO on the OneLeg2DEnv custom environment"""
    
    current_time_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    frame_per_batch = int(train_cfg["frame_per_batch"])
    sub_batch_size = int(train_cfg["sub_batch_size"])
    log_interval_base_num = int(train_cfg["log_interval_base_num"])
    num_envs = int(train_cfg["num_envs"])

    # Eval config (required)
    eval_cfg = _config["eval"]

    agent = PPOTorchRLSymmetric(
        # env_class=OneLeg2DEnv,  # Use custom environment
        env_class=env_class,  # Use custom environment
        env_kwargs=env_kwargs,
        total_frames=total_timesteps,
        frames_per_batch=frame_per_batch,
        sub_batch_size=sub_batch_size,
        lr=float(ppo_cfg["lr"]),
        clip_epsilon=float(ppo_cfg["clip_epsilon"]),
        gamma=float(ppo_cfg["gamma"]),
        max_grad_norm=float(ppo_cfg["max_grad_norm"]),
        target_kl=float(ppo_cfg["target_kl"]),
        initial_scale=float(ppo_cfg["initial_scale"]),
        scale_min=float(ppo_cfg["scale_min"]),
        scale_max=float(ppo_cfg["scale_max"]),
        scale_distribution_method=str(ppo_cfg["scale_distribution_method"]),
        use_obs_norm=bool(ppo_cfg.get("use_obs_norm", False)),

        # for network
        actor_architecture=[tuple(layer) for layer in ppo_cfg["actor_architecture"]],
        critic_architecture=[tuple(layer) for layer in ppo_cfg["critic_architecture"]],
        actor_net_init_gain=float(ppo_cfg["actor_net_init_gain"]),
        actor_net_init_bias=float(ppo_cfg["actor_net_init_bias"]),
        critic_net_init_gain=float(ppo_cfg["critic_net_init_gain"]),
        critic_net_init_bias=float(ppo_cfg["critic_net_init_bias"]),
        actor_weight_init_method=str(ppo_cfg["actor_weight_init_method"]),
        critic_weight_init_method=str(ppo_cfg["critic_weight_init_method"]),

        log_interval=train_cfg["log_interval"],
        eval_interval=train_cfg["eval_interval"],
        eval_num_episodes=eval_cfg["num_episodes"],
        eval_max_episode_length=eval_cfg["max_episode_length"],

        num_envs=num_envs,
        base_dir=os.path.join(
            logging_cfg["base_dir"],
            f"{current_time_str}_{env_kwargs['env_id']}_{symmetric_method}"
        ),

        seed=train_cfg["seed"],


        info_to_log={
            # "symmetric_method": symmetric_method,
        },

        symmetric_method = symmetric_method,
        symmetric_params = symmetric_params,
    )
    
    # Train the agent
    print("\nStarting training...")
    logs = agent.train()
    
    # Evaluate the trained agent
    print("\nEvaluating trained agent...")
    eval_results = agent.evaluate(num_episodes=3,
                                  evaluate_name="last_trained",
                                  render_frames=True
                                  )