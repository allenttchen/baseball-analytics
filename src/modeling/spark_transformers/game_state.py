from itertools import chain
from typing import List

from pyspark.ml import Transformer
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql.functions import (
    col,
    lit,
    create_map,
    year, to_date,
    datediff,
    when,
    to_timestamp,
)
from pyspark.sql import DataFrame

from src.modeling.utils import time_job
from src.modeling.constants import SEASON_START_DATES


class GetOutputColsMixin:
    def get_output_cols(self) -> List[str]:
        if hasattr(self, "outputCol"):
            return [self.outputCol]
        elif hasattr(self, "outputCols"):
            return self.outputCols
        else:
            return []


class ConvertToDatetime(
    Transformer,
    DefaultParamsReadable,
    DefaultParamsWritable,
    GetOutputColsMixin,
):
    def __init__(self, inputCol=None, outputCol=None):
        super(ConvertToDatetime, self).__init__()
        self.inputCol = inputCol
        self.outputCol = outputCol

    def _transform(self, dataset: DataFrame) -> DataFrame:
        dataset = dataset.withColumn(
            self.outputCol,
            to_timestamp(col(self.inputCol), "yyyy-MM-dd")
        )
        return dataset


class EncodeOnBaseOccupancy(
    Transformer,
    DefaultParamsReadable,
    DefaultParamsWritable,
    GetOutputColsMixin,
):

    def __init__(self, inputCols=None, outputCols=None):
        super(EncodeOnBaseOccupancy, self).__init__()
        self.inputCols = inputCols
        self.outputCols = outputCols

    @time_job
    def _transform(self, dataset: DataFrame) -> DataFrame:
        for input_col, output_col in zip(self.inputCols, self.outputCols):
            dataset = dataset.withColumn(
                output_col,
                when(col(input_col) != 0, 1).otherwise(0),
            )
        return dataset


class ComputeNetScore(
    Transformer,
    DefaultParamsReadable,
    DefaultParamsWritable,
    GetOutputColsMixin,
):
    def __init__(self, inputCols=None, outputCol=None):
        super(ComputeNetScore, self).__init__()
        self.inputCols = inputCols
        self.outputCol = outputCol

    @time_job
    def _transform(self, dataset: DataFrame) -> DataFrame:
        dataset = dataset.withColumn(
            self.outputCol,
            col(self.inputCols[0]) - col(self.inputCols[1])
        )
        return dataset


class ComputeDaysSinceStart(
    Transformer,
    DefaultParamsReadable,
    DefaultParamsWritable,
    GetOutputColsMixin,
):
    def __init__(self, inputCol=None, outputCol=None):
        super(ComputeDaysSinceStart, self).__init__()
        self.inputCol = inputCol
        self.outputCol = outputCol
        self.season_start_dates = create_map([lit(x) for x in chain(*SEASON_START_DATES.items())])

    @time_job
    def _transform(self, dataset: DataFrame) -> DataFrame:
        dataset = dataset.withColumn(
            self.outputCol,
            datediff(
                to_date(col(self.inputCol)),
                to_date(self.season_start_dates[year(col(self.inputCol))], "yyyy-MM-dd"),
            )
        )
        return dataset


class EncodeHandedness(
    Transformer,
    DefaultParamsReadable,
    DefaultParamsWritable,
    GetOutputColsMixin,
):
    def __init__(self, inputCols=None, outputCols=None):
        super(EncodeHandedness, self).__init__()
        self.inputCols = inputCols
        self.outputCols = outputCols
        self.handedness_mapping = create_map([lit(x) for x in chain(*dict(R=1, L=0).items())])

    @time_job
    def _transform(self, dataset: DataFrame) -> DataFrame:
        for input_col, output_col in zip(self.inputCols, self.outputCols):
            dataset = dataset.withColumn(
                output_col,
                self.handedness_mapping[col(input_col)]
            )
        return dataset


class EncodeInningTopBot(
    Transformer,
    DefaultParamsReadable,
    DefaultParamsWritable,
    GetOutputColsMixin,
):
    def __init__(self, inputCol=None, outputCol=None):
        super(EncodeInningTopBot, self).__init__()
        self.inputCol = inputCol
        self.outputCol = outputCol
        self.inningtopbot_mapping = create_map([lit(x) for x in chain(*dict(Top=1, Bot=0).items())])

    @time_job
    def _transform(self, dataset: DataFrame) -> DataFrame:
        dataset = dataset.withColumn(
            self.outputCol,
            self.inningtopbot_mapping[col(self.inputCol)]
        )
        return dataset
