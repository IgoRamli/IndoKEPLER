from transformers import Trainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl

class KeplerCallback(TrainerCallback):
    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.tr_mlm_loss = 0
        self.tr_ke_loss = 0
        self.eval_mlm_loss = 0
        self.eval_ke_loss = 0
        return super().on_train_begin(args, state, control, **kwargs)
        

class KeplerTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
            
        loss = outputs["loss"]

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs):
        # We removed SageMaker and AMP support
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

        self.tr_mlm_loss += outputs["mlm_loss"]
        self.tr_ke_loss += outputs["ke_loss"]

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()

        return loss.detach()

    def log(self, logs):
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        other_logs = {
            "step": self.state.global_step,
            "mlm_loss": self.tr_mlm_loss,
            "ke_loss": self.tr_ke_loss
        }
        output = {**logs, **other_logs}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)