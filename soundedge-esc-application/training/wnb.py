# --------------------------------------------------------------------------- #
# Weights & Biases (no-op when disabled, so callers stay branch-free)
# --------------------------------------------------------------------------- #
class WandbLogger:
    def __init__(self, args):
        self.enabled = args.wandb
        if not self.enabled:
            return
        import wandb

        self._wandb = wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run,
            config=vars(args),
        )

    def log(self, metrics):
        if self.enabled:
            self._wandb.log(metrics)

    def summarize(self, **kv):
        if self.enabled:
            for k, v in kv.items():
                if v is not None:
                    self._wandb.run.summary[k] = v

    def finish(self):
        if self.enabled:
            self._wandb.finish()
