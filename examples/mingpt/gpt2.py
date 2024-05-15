import hydra
from omegaconf import DictConfig
from torch.distributed import destroy_process_group

import gpt
import predictor
from char_dataset import DataConfig
from model import GPTConfig, OptimizerConfig
from trainer import Trainer, TrainerConfig


@predictor.entrypoint(max_analyze_depth=100, min_tracing_possibility=0.5, max_tracing_time=3.0, min_report_time=1.0, max_report_time=3.0)
@predictor.rules.enable_common_rules
@predictor.rules.enable_pytorch_rules
@hydra.main(version_base=None, config_path=".", config_name="gpt2_train_cfg")
def main(cfg: DictConfig):
    gpt.ddp_setup()

    gpt_cfg = GPTConfig(**cfg['gpt_config'])
    opt_cfg = OptimizerConfig(**cfg['optimizer_config'])
    data_cfg = DataConfig(**cfg['data_config'])
    trainer_cfg = TrainerConfig(**cfg['trainer_config'])

    model, optimizer, train_data, test_data = gpt.get_train_objs(gpt_cfg, opt_cfg, data_cfg)
    trainer = Trainer(trainer_cfg, model, optimizer, train_data, test_data)
    trainer.train()

    destroy_process_group()


if __name__ == "__main__":
    main()
