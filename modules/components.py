"""Initiate tfx pipeline components
"""

import os

import tensorflow_model_analysis as tfma
from tfx.components import (
    CsvExampleGen,
    StatisticsGen,
    SchemaGen,
    ExampleValidator,
    Transform,
    Trainer,
    Tuner,
    Evaluator,
    Pusher
)
from tfx.proto import example_gen_pb2, trainer_pb2, pusher_pb2
from tfx.types import Channel
from tfx.dsl.components.common.resolver import Resolver
from tfx.types.standard_artifacts import Model, ModelBlessing
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import (
    LatestBlessedModelStrategy)


def init_components(args):
    """Initiate tfx pipeline components

    Args:
        args (dict): a dictionary of data_dir, trainer_module, tuner_module, transform_module, train_steps, eval_steps, serving_model_dir related to data
    Returns:
        TFX components
    """
    output = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(splits=[
            example_gen_pb2.SplitConfig.Split(name="train", hash_buckets=8),
            example_gen_pb2.SplitConfig.Split(name="eval", hash_buckets=2)
        ])
    )

    # konfigurasi output
    example_gen = CsvExampleGen(
        input_base=args["data_dir"],
        output_config=output
    )

    # memvalidasi data
    statistics_gen = StatisticsGen(
        examples=example_gen.outputs["examples"]
    )

    # memvalidasi data
    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs["statistics"]
    )

    # memvalidasi data
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema']
    )

    # data preprocessing
    transform = Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=os.path.abspath(args["transform_module"])
    )

    # hyperparameter tuning untuk proses training
    tuner = Tuner(
        module_file=os.path.abspath(args["tuner_module"]),
        examples=transform.outputs["transformed_examples"],
        transform_graph=transform.outputs["transform_graph"],
        schema=schema_gen.outputs["schema"],
        train_args=trainer_pb2.TrainArgs(
            splits=["train"],
            num_steps=args["train_steps"],
        ),
        eval_args=trainer_pb2.EvalArgs(
            splits=["eval"],
            num_steps=args["eval_steps"],
        ),
    )

    # menjalankan proses training
    trainer = Trainer(
        module_file=os.path.abspath(args["trainer_module"]),
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        hyperparameters=tuner.outputs["best_hyperparameters"],
        train_args=trainer_pb2.TrainArgs(
            splits=['train'],
            num_steps=args["train_steps"]),
        eval_args=trainer_pb2.EvalArgs(
            splits=['eval'],
            num_steps=args["eval_steps"])
    )

    # menyiapkan sebuah baseline model
    model_resolver = Resolver(
        strategy_class=LatestBlessedModelStrategy,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing)
    ).with_id('Latest_blessed_model_resolver')

    # menjalankan proses analisis dan validasi model
    slicing_specs = [
        tfma.SlicingSpec(),
        tfma.SlicingSpec(feature_keys=[
            "gender",
            "hypertension",
            "heart_disease"
        ])
    ]

    # menjalankan proses analisis dan validasi model
    metrics_specs = [
        tfma.MetricsSpec(metrics=[
            tfma.MetricConfig(class_name='AUC'),
            tfma.MetricConfig(class_name="Precision"),
            tfma.MetricConfig(class_name="Recall"),
            tfma.MetricConfig(class_name="ExampleCount"),
            tfma.MetricConfig(class_name='BinaryAccuracy',
                              threshold=tfma.MetricThreshold(
                                  value_threshold=tfma.GenericValueThreshold(
                                      lower_bound={'value': 0.5}),
                                  change_threshold=tfma.GenericChangeThreshold(
                                      direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                                      absolute={'value': 0.0001})
                              )
                              )
        ])
    ]

    # menjalankan proses analisis dan validasi model
    eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key='stroke')],
        slicing_specs=slicing_specs,
        metrics_specs=metrics_specs
    )

    # menjalankan proses analisis dan validasi model
    evaluator = Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        baseline_model=model_resolver.outputs['model'],
        eval_config=eval_config)

    # menyiapkan model yang akan masuk ke sistem produksi
    pusher = Pusher(
        model=trainer.outputs["model"],
        model_blessing=evaluator.outputs["blessing"],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=args["serving_model_dir"]
            )
        ),
    )

    return (
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        tuner,
        trainer,
        model_resolver,
        evaluator,
        pusher
    )
