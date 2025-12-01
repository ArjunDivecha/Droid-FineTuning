import argparse
import math
import os
import re
import types
import warnings
from pathlib import Path
import time
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import yaml
from mlx.utils import tree_flatten, tree_map
from mlx.nn.utils import average_gradients

# Imports from mlx_lm
from mlx_lm.tuner.callbacks import get_reporting_callbacks, TrainingCallback
from mlx_lm.tuner.datasets import CacheDataset, load_dataset
from mlx_lm.tuner.trainer import TrainingArgs, evaluate, default_loss, iterate_batches, grad_checkpoint
from mlx_lm.tuner.utils import (
    build_schedule,
    linear_to_lora_layers,
    load_adapters,
    print_trainable_parameters,
)
from mlx_lm.utils import load, save_config

yaml_loader = yaml.SafeLoader
yaml_loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        """^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)

CONFIG_DEFAULTS = {
    "model": "Qwen/Qwen3-0.6b",
    "train": False,
    "fine_tune_type": "lora",
    "optimizer": "adam",
    "optimizer_config": {
        "adam": {},
        "adamw": {},
        "muon": {},
        "sgd": {},
        "adafactor": {},
    },
    "data": "mlx-community/WikiSQL",
    "seed": 0,
    "num_layers": 16,
    "batch_size": 4,
    "iters": 1000,
    "val_batches": 25,
    "learning_rate": 1e-5,
    "steps_per_report": 10,
    "steps_per_eval": 200,
    "resume_adapter_file": None,
    "adapter_path": "adapters",
    "save_every": 100,
    "test": False,
    "test_batches": 500,
    "max_seq_length": 2048,
    "config": None,
    "grad_checkpoint": False,
    "grad_accumulation_steps": 1,
    "lr_schedule": None,
    "lora_parameters": {"rank": 8, "dropout": 0.0, "scale": 20.0},
    "mask_prompt": False,
    "report_to": None,
    "project_name": None,
}

# COPIED AND FIXED TRAIN FUNCTION
def train(
    model,
    optimizer,
    train_dataset,
    val_dataset,
    args: TrainingArgs = TrainingArgs(),
    loss: callable = default_loss,
    iterate_batches: callable = iterate_batches,
    training_callback: TrainingCallback = None,
):
    if mx.metal.is_available():
        mx.set_wired_limit(mx.metal.device_info()["max_recommended_working_set_size"])
    print(f"Starting training..., iters: {args.iters}")
    world = mx.distributed.init()
    world_size = world.size()
    rank = world.rank()
    if world_size > 1:
        print(f"Node {rank} of {world_size}")

    if args.grad_checkpoint:
        grad_checkpoint(model.layers[0])

    loss_value_and_grad = nn.value_and_grad(model, loss)

    grad_accum_steps = args.grad_accumulation_steps
    if grad_accum_steps < 1:
        raise ValueError("grad_accumulation_steps must be at least 1")

    state = [model.state, optimizer.state, mx.random.state]

    # @partial(mx.compile, inputs=state, outputs=state)
    def step(batch, prev_grad, do_update):
        (lvalue, toks), grad = loss_value_and_grad(model, *batch)

        if prev_grad is not None:
            grad = tree_map(lambda x, y: x + y, grad, prev_grad)

        if do_update:
            grad = average_gradients(grad)
            if grad_accum_steps > 1:
                grad = tree_map(lambda x: x / grad_accum_steps, grad)
            optimizer.update(model, grad)
            grad = None

        return lvalue, toks, grad

    model.train()
    losses = 0
    n_tokens = 0
    steps = 0
    trained_tokens = 0
    train_time = 0
    grad_accum = None
    
    final_train_loss = None
    final_val_loss = None

    # Main training loop
    for it, batch in zip(
        range(1, args.iters + 1),
        iterate_batches(
            dataset=train_dataset,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            train=True,
        ),
    ):
        tic = time.perf_counter()
        # Report validation loss if needed, the first validation loss
        # is always measured before any training.
        if it == 1 or it % args.steps_per_eval == 0 or it == args.iters:
            tic = time.perf_counter()
            val_loss = evaluate(
                model=model,
                dataset=val_dataset,
                loss=loss,
                batch_size=args.batch_size,
                num_batches=args.val_batches,
                max_seq_length=args.max_seq_length,
                iterate_batches=iterate_batches,
            )
            model.train()
            val_time = time.perf_counter() - tic
            if rank == 0:
                print(
                    f"Iter {it}: "
                    f"Val loss {val_loss:.3f}, "
                    f"Val took {val_time:.3f}s",
                    flush=True,
                )
            final_val_loss = float(val_loss)

            if training_callback is not None:
                val_info = {
                    "iteration": it - 1,
                    "val_loss": val_loss,
                    "val_time": val_time,
                }
                training_callback.on_val_loss_report(val_info)

            tic = time.perf_counter()

        lvalue, toks, grad_accum = step(
            batch,
            grad_accum,
            it % grad_accum_steps == 0,
        )

        losses += lvalue
        n_tokens += toks
        steps += 1
        
        # FIX: Flatten state and grad_accum to get list of arrays
        # tree_flatten returns list of (path, value) tuples, we need just values
        flat_state = [v for _, v in tree_flatten(state)]
        flat_grad = [v for _, v in tree_flatten(grad_accum)] if grad_accum is not None else []
        
        # Evaluate everything by unpacking the lists
        mx.eval(*flat_state, losses, n_tokens, *flat_grad)
        
        # AGGRESSIVE CLEANUP
        if it % 10 == 0:
            mx.metal.clear_cache()
            import gc
            gc.collect()
        
        train_time += time.perf_counter() - tic

        # Report training loss if needed
        if it % args.steps_per_report == 0 or it == args.iters:
            train_loss = mx.distributed.all_sum(losses, stream=mx.cpu).item()
            train_loss /= steps * world_size
            n_tokens = mx.distributed.all_sum(n_tokens, stream=mx.cpu).item()
            learning_rate = optimizer.learning_rate.item()
            it_sec = args.steps_per_report / train_time
            tokens_sec = float(n_tokens) / train_time
            trained_tokens += n_tokens
            peak_mem = mx.get_peak_memory() / 1e9
            if rank == 0:
                print(
                    f"Iter {it}: Train loss {train_loss:.3f}, "
                    f"Learning Rate {learning_rate:.3e}, "
                    f"It/sec {it_sec:.3f}, "
                    f"Tokens/sec {tokens_sec:.3f}, "
                    f"Trained Tokens {trained_tokens}, "
                    f"Peak mem {peak_mem:.3f} GB",
                    flush=True,
                )
            final_train_loss = float(train_loss)

            if training_callback is not None:
                train_info = {
                    "iteration": it,
                    "train_loss": train_loss,
                    "learning_rate": learning_rate,
                    "iterations_per_second": it_sec,
                    "tokens_per_second": tokens_sec,
                    "trained_tokens": trained_tokens,
                    "peak_memory": peak_mem,
                }
                training_callback.on_train_loss_report(train_info)

            losses = 0
            n_tokens = 0
            steps = 0
            train_time = 0

        # Save adapter weights
        if it % args.steps_per_save == 0 and rank == 0:
            adapter_weights = dict(tree_flatten(model.trainable_parameters()))
            mx.save_safetensors(str(args.adapter_file), adapter_weights)
            checkpoint = (
                Path(args.adapter_file).parent / f"{it:07d}_adapters.safetensors"
            )
            mx.save_safetensors(str(checkpoint), adapter_weights)
            print(
                f"Iter {it}: Saved adapter weights to "
                f"{args.adapter_file} and {checkpoint}."
            )

    # Save final weights
    if rank == 0:
        adapter_weights = dict(tree_flatten(model.trainable_parameters()))
        mx.save_safetensors(str(args.adapter_file), adapter_weights)
        print(f"Saved final weights to {args.adapter_file}.")

    return {
        'final_train_loss': final_train_loss,
        'final_val_loss': final_val_loss
    }


def build_parser():
    parser = argparse.ArgumentParser(description="LoRA or QLoRA finetuning.")
    parser.add_argument(
        "--model",
        type=str,
        help="The path to the local model directory or Hugging Face repo.",
    )

    # Training args
    parser.add_argument(
        "--train",
        action="store_true",
        help="Do training",
        default=None,
    )
    parser.add_argument(
        "--data",
        type=str,
        help=(
            "Directory with {train, valid, test}.jsonl files or the name "
            "of a Hugging Face dataset (e.g., 'mlx-community/wikisql')"
        ),
    )
    parser.add_argument(
        "--fine-tune-type",
        type=str,
        choices=["lora", "dora", "full"],
        help="Type of fine-tuning to perform: lora, dora, or full.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adam", "adamw", "muon", "sgd", "adafactor"],
        default=None,
        help="Optimizer to use for training: adam, adamw, sgd, or adafactor.",
    )
    parser.add_argument(
        "--mask-prompt",
        action="store_true",
        help="Mask the prompt in the loss when training",
        default=None,
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        help="Number of layers to fine-tune. Default is 16, use -1 for all.",
    )
    parser.add_argument("--batch-size", type=int, help="Minibatch size.")
    parser.add_argument("--iters", type=int, help="Iterations to train for.")
    parser.add_argument(
        "--val-batches",
        type=int,
        help="Number of validation batches, -1 uses the entire validation set.",
    )
    parser.add_argument("--learning-rate", type=float, help="Adam learning rate.")
    parser.add_argument(
        "--steps-per-report",
        type=int,
        help="Number of training steps between loss reporting.",
    )
    parser.add_argument(
        "--steps-per-eval",
        type=int,
        help="Number of training steps between validations.",
    )
    parser.add_argument(
        "--grad-accumulation-steps",
        type=int,
        help="Number of steps to accumulate before each optimizer update.",
    )
    parser.add_argument(
        "--resume-adapter-file",
        type=str,
        help="Load path to resume training from the given fine-tuned weights.",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        help="Save/load path for the fine-tuned weights.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        help="Save the model every N iterations.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Evaluate on the test set after training",
        default=None,
    )
    parser.add_argument(
        "--test-batches",
        type=int,
        help="Number of test set batches, -1 uses the entire test set.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        help="Maximum sequence length.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="A YAML configuration file with the training options",
    )
    parser.add_argument(
        "--grad-checkpoint",
        action="store_true",
        help="Use gradient checkpointing to reduce memory use.",
        default=None,
    )
    parser.add_argument(
        "--report-to",
        type=str,
        default=None,
        help="Services to report logs to ('wandb', 'swanlab', or 'wandb,swanlab').",
    )
    parser.add_argument(
        "--project-name",
        type=str,
        default=None,
        help="Project name for logging. Defaults to the name of the root directory.",
    )
    parser.add_argument("--seed", type=int, help="The PRNG seed")
    return parser


def train_model(
    args,
    model: nn.Module,
    train_set,
    valid_set,
    training_callback: TrainingCallback = None,
):
    mx.random.seed(args.seed)
    model.freeze()
    if args.num_layers > len(model.layers):
        raise ValueError(
            f"Requested to train {args.num_layers} layers "
            f"but the model only has {len(model.layers)} layers."
        )

    if args.fine_tune_type == "full":
        for l in model.layers[-max(args.num_layers, 0) :]:
            l.unfreeze()

        args.lora_parameters = None
    elif args.fine_tune_type in ["lora", "dora"]:
        # Convert linear layers to lora/dora layers and unfreeze in the process
        linear_to_lora_layers(
            model,
            args.num_layers,
            args.lora_parameters,
            use_dora=(args.fine_tune_type == "dora"),
        )
    else:
        raise ValueError(f"Received unknown fine-tune-type {args.fine_tune_type}")

    # Resume from weights if provided
    if args.resume_adapter_file is not None:
        print(f"Loading fine-tuned weights from {args.resume_adapter_file}")
        model.load_weights(args.resume_adapter_file, strict=False)

    print_trainable_parameters(model)

    adapter_path = Path(args.adapter_path)
    adapter_path.mkdir(parents=True, exist_ok=True)

    adapter_file = adapter_path / "adapters.safetensors"
    save_config(vars(args), adapter_path / "adapter_config.json")

    # init training args
    training_args = TrainingArgs(
        batch_size=args.batch_size,
        iters=args.iters,
        val_batches=args.val_batches,
        steps_per_report=args.steps_per_report,
        steps_per_eval=args.steps_per_eval,
        steps_per_save=args.save_every,
        adapter_file=adapter_file,
        max_seq_length=args.max_seq_length,
        grad_checkpoint=args.grad_checkpoint,
        grad_accumulation_steps=args.grad_accumulation_steps,
    )

    # Initialize the selected optimizer
    lr = build_schedule(args.lr_schedule) if args.lr_schedule else args.learning_rate

    optimizer_name = args.optimizer.lower()
    optimizer_config = args.optimizer_config.get(optimizer_name, {})
    if optimizer_name == "adam":
        opt_class = optim.Adam
    elif optimizer_name == "adamw":
        opt_class = optim.AdamW
    elif optimizer_name == "muon":
        opt_class = optim.Muon
    elif optimizer_name == "sgd":
        opt_class = optim.SGD
    elif optimizer_name == "adafactor":
        opt_class = optim.Adafactor
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    opt = opt_class(learning_rate=lr, **optimizer_config)

    # Train model
    metrics = train(
        model=model,
        args=training_args,
        optimizer=opt,
        train_dataset=CacheDataset(train_set),
        val_dataset=CacheDataset(valid_set),
        training_callback=training_callback,
    )

    # Save final metrics to config
    if metrics:
        print(f"Saving final metrics to {adapter_path / 'adapter_config.json'}")
        with open(adapter_path / "adapter_config.json", "r") as f:
            config = json.load(f)
        
        config.update(metrics)
        
        with open(adapter_path / "adapter_config.json", "w") as f:
            json.dump(config, f, indent=4)


def evaluate_model(args, model: nn.Module, test_set):
    test_loss = evaluate(
        model=model,
        dataset=CacheDataset(test_set),
        batch_size=args.batch_size,
        num_batches=args.test_batches,
        max_seq_length=args.max_seq_length,
    )

    test_ppl = math.exp(test_loss)

    print(f"Test loss {test_loss:.3f}, Test ppl {test_ppl:.3f}.")


def run(args, training_callback: TrainingCallback = None):
    np.random.seed(args.seed)
    training_callback = get_reporting_callbacks(
        args.report_to,
        project_name=args.project_name,
        log_dir=args.adapter_path,
        config=vars(args),
    )

    print("Loading pretrained model")
    model, tokenizer = load(args.model, tokenizer_config={"trust_remote_code": True})

    print("Loading datasets")
    train_set, valid_set, test_set = load_dataset(args, tokenizer)

    if args.test and not args.train:
        # Allow testing without LoRA layers by providing empty path
        if args.adapter_path != "":
            load_adapters(model, args.adapter_path)

    elif args.train:
        print("Training")
        train_model(args, model, train_set, valid_set, training_callback)
    else:
        raise ValueError("Must provide at least one of --train or --test")

    if args.test:
        print("Testing")
        evaluate_model(args, model, test_set)


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    parser = build_parser()
    args = parser.parse_args()
    config = args.config
    args = vars(args)
    if config:
        print("Loading configuration file", config)
        with open(config, "r") as file:
            config = yaml.load(file, yaml_loader)
        # Prefer parameters from command-line arguments
        for k, v in config.items():
            if args.get(k, None) is None:
                args[k] = v

    # Update defaults for unspecified parameters
    for k, v in CONFIG_DEFAULTS.items():
        if args.get(k, None) is None:
            args[k] = v
    run(types.SimpleNamespace(**args))


if __name__ == "__main__":
    main()
