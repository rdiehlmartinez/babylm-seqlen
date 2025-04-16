def get_deepspeed_config():
    return {
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {"device": "cpu"},
            "offload_param": {"device": "cpu"},
        },
        "train_batch_size": 64,
        "gradient_accumulation_steps": 1,
        "fp16": {"enabled": True},
    }
