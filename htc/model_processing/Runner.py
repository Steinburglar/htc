# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import argparse
import copy
import multiprocessing
import queue
import sys
from pathlib import Path

import psutil
import torch

from htc.model_processing.ImageConsumer import ImageConsumer
from htc.model_processing.Predictor import Predictor
from htc.models.common.HTCModel import HTCModel
from htc.settings import settings


class Runner:
    def __init__(self, description: str) -> None:
        r"""
        Helper class to start the producer and the consumers to operate on predictions of images.

        The runner defines same basic arguments which apply to all scripts (e.g. --model and --run-folder to define the trained model). However, scripts can also add their own arguments. Please use the add_argument() method for this purpose. You can use it in the same way as with parser.add_argument() but it already defines some predefined arguments which are needed in multiple scripts.

        Default arguments:
        >>> import re
        >>> runner = Runner(description="My inference script")
        >>> re.sub(r'\s+', ' ', runner.parser.format_usage())  # doctest: +ELLIPSIS
        'usage: ... [-h] --model MODEL --run-folder RUN_FOLDER [--num-consumers NUM_CONSUMERS] [--store-predictions] [--use-predictions] [--hide-progressbar] '

        Adding a custom argument with default options:
        >>> runner.add_argument("--input-dir")
        >>> re.sub(r'\s+', ' ', runner.parser.format_usage())  # doctest: +ELLIPSIS
        'usage: ... [-h] --model MODEL --run-folder RUN_FOLDER [--num-consumers NUM_CONSUMERS] [--store-predictions] [--use-predictions] [--hide-progressbar] [--input-dir INPUT_DIR] '

        Custom argument with a different option (note that --input-dir is now a required argument):
        >>> runner = Runner(description="My inference script")
        >>> runner.parser.formatter_class = argparse.RawDescriptionHelpFormatter
        >>> runner.add_argument("--input-dir", required=True)
        >>> re.sub(r'\s+', ' ', runner.parser.format_usage())  # doctest: +ELLIPSIS
        'usage: ... [-h] --model MODEL --run-folder RUN_FOLDER [--num-consumers NUM_CONSUMERS] [--store-predictions] [--use-predictions] [--hide-progressbar] --input-dir INPUT_DIR '

        All custom arguments are automatically passed to the producer and the consumers, i.e. self.input_dir contains the path which the user submitted as command line argument.

        Args:
            description: Short description of what the script is doing. Will be added to the help message.
        """
        self.parser = argparse.ArgumentParser(
            description=(
                "Calculate the inference for a trained (potentially be downloaded) model for a set of images. Script"
                f" description: {description}"
            ),
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        self.parser.add_argument(
            "--model",
            type=str,
            required=True,
            help="Name of the model to train (e.g. image or pixel).",
        )
        self.parser.add_argument(
            "--run-folder",
            type=str,
            required=True,
            help=(
                "The name of the directory which stores the training results (e.g."
                " 2022-02-03_22-58-44_generated_default_model_comparison)."
            ),
        )
        self.parser.add_argument(
            "--num-consumers",
            type=int,
            default=None,
            help=(
                "Number of consumers/processes to spawn which work on the predicted images. Defaults to n_physical_cpus"
                " - 1 to have at least one free CPU for the inference. Note that sometimes less is more and using fewer"
                " consumers may speed up the program time (e.g. if your consumer code uses many threads)."
            ),
        )
        self.parser.add_argument(
            "--store-predictions",
            default=False,
            action="store_true",
            help=(
                "Store predictions (<image_name>.blosc file with softmax predictions). If a script (or another script)"
                " is run on the same run directory again and the --use-predictions switch is set, then the"
                " precalculated predictions are used per default."
            ),
        )
        self.parser.add_argument(
            "--use-predictions",
            default=False,
            action="store_true",
            help="Use existing predictions if they already exist.",
        )
        self.parser.add_argument(
            "--hide-progressbar",
            default=False,
            action="store_true",
            help="If set, no progress bar is shown for the predictor.",
        )

        self._args = None
        self._used_args = []

    @property
    def args(self) -> argparse.Namespace:
        if self._args is None:
            self._args = self.parser.parse_args()

        return self._args

    @property
    def run_dir(self) -> Path:
        return HTCModel.find_pretrained_run(self.args.model, self.args.run_folder)

    def add_argument(self, name: str, **kwargs) -> None:
        """
        Add a custom argument to the runner. If the name is known (e.g. --test), then some defaults will be applied to the argument. It is always possible to overwrite the defaults. The value of all arguments is automatically passed on to the producer and the consumers.

        Args:
            name: Name of the argument (e.g. --test).
            kwargs: All additional keyword arguments passed to parser.add_argument()
        """
        if name == "--test":
            kwargs.setdefault("default", False)
            kwargs.setdefault("action", "store_true")
            kwargs.setdefault("help", "Create predictions on the test set (using the TestPredictor class).")
        elif name == "--test-looc":
            kwargs.setdefault("default", False)
            kwargs.setdefault("action", "store_true")
            kwargs.setdefault("help", "Use the TestLeaveOneOutPredictor class for test predictions (no ensembling).")
        elif name == "--input-dir":
            kwargs.setdefault("type", Path)
            kwargs.setdefault("default", None)
            kwargs.setdefault("help", "The directory path containing input data for the script.")
        elif name == "--output-dir":
            kwargs.setdefault("type", Path)
            kwargs.setdefault("default", None)
            kwargs.setdefault("help", "Output directory where the generated files should be stored.")
        elif name == "--metrics":
            kwargs.setdefault("type", str)
            kwargs.setdefault("nargs", "*")
            kwargs.setdefault("default", ["DSC", "ASD", "NSD"])
            kwargs.setdefault(
                "help",
                (
                    "The metrics which are to be calculated. By default, the dice, average surface distance and"
                    " normalized surface dice are used."
                ),
            )
        elif name == "--fold-name":
            kwargs.setdefault("type", str)
            kwargs.setdefault("default", None)
            kwargs.setdefault(
                "help",
                (
                    "The name of the fold for which the activations have to be calculated. Currently activations"
                    " plotting is only implemented for the hyper_diva model."
                ),
            )
        elif name == "--target-domain":
            kwargs.setdefault("type", str)
            kwargs.setdefault("default", None)
            kwargs.setdefault(
                "help",
                (
                    "The target domain for hyper_diva activations plotting. If this parameter is specified then the"
                    " target_domain from the config files is overridden."
                ),
            )

        self.parser.add_argument(name, **kwargs)
        self._used_args.append(name.removeprefix("--").replace("-", "_"))

    def start(self, PredictorClass: type[Predictor], ConsumerClass: type[ImageConsumer]) -> None:
        """
        Start the producer and the consumers. If you need to pass additional parameters (not CLI arguments, they are automatically available in the producer and the consumers) to your consumers or producer, please use the functools.partial method.

        Args:
            PredictorClass: Class type (not an instance) used to create image predictions (usually TestPredictor or ValidationPredictor).
            ConsumerClass: Your consumer class type (not an instance).
        """
        with multiprocessing.Manager() as manager:
            # We provide two options per default to share results across consumers
            results_list = manager.list()
            results_dict = manager.dict()

            # Start consumers which will work on the image predictions
            if self.args.num_consumers is None:
                num_consumers = psutil.cpu_count(logical=False) - 1  # The main process will calculate the predictions
            else:
                num_consumers = self.args.num_consumers
            task_queue = manager.JoinableQueue(maxsize=num_consumers)
            task_queue_errors = manager.Queue(maxsize=num_consumers)

            # All arguments added via add_argument will be passed to the producer and the consumer
            additional_kwargs = {name: getattr(self.args, name) for name in self._used_args}

            # specify all arguments with explicit names for ConsumerClass to avoid errors due to partial class initialization
            consumer = ConsumerClass(
                task_queue=task_queue,
                task_queue_errors=task_queue_errors,
                results_list=results_list,
                results_dict=results_dict,
                run_dir=self.run_dir,
                store_predictions=self.args.store_predictions,
                **additional_kwargs,
            )
            consumers = [
                copy.copy(consumer) for _ in range(num_consumers)
            ]  # A copy is much cheaper if __init__ is expensive
            for w in consumers:
                w.start()

            # The producer creates the predictions (only the main process since we use only 1 GPU)
            predictor = PredictorClass(
                self.run_dir, self.args.use_predictions, self.args.store_predictions, **additional_kwargs
            )

            exit_code = 0
            try:
                with torch.autocast(device_type="cuda"):
                    predictor.start(task_queue, self.args.hide_progressbar)

                task_queue.join()  # Wait until all consumers are finished with this run
                task_queue.put(
                    self.run_dir
                )  # Let one consumer do the final work (the run_dir serves as a dummy to indicate that all predictions are done)
            except Exception:
                settings.log.exception("Error occurred in the producer")
                exit_code = 1
            finally:
                # Add a poison pill for each consumer to shut down
                for _ in range(num_consumers):
                    task_queue.put(None)

                task_queue.join()

            # Check whether some errors have happened
            try:
                # Consumers always return error information
                errors = []
                for _ in range(num_consumers):
                    response = task_queue_errors.get(timeout=30)  # timeout in seconds
                    if response is not None:
                        errors.append(response)
            except queue.Empty:
                exit_code = 1
                settings.log.error("Did not receive an answer about errors from every consumer")

            if len(errors) > 0:
                msg = (
                    f'{len(errors)} consumer{"s" if len(errors) > 1 else ""} failed with an error for the run'
                    f" {self.run_dir}:\n"
                )
                msg += "\n".join([str(e) for e in errors])
                settings.log.error(msg)
                exit_code = 1

            sys.exit(exit_code)