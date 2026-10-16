"""Data contract for time series in long format."""

import pandera.polars as pa
import polars as pl


class SeriesSchema(pa.DataFrameModel):
    """One row = one observation of one series at one timestamp."""

    series_id: str = pa.Field(nullable=False)
    timestamp: pl.Datetime = pa.Field(nullable=False)
    y: float = pa.Field(ge=0, nullable=False)  # counts cannot be negative

    class Config:
        strict = False  # TODO: decide. True would reject unknown columns (covariates)
        coerce = True  # cast to the declared types when possible
        unique = ["series_id", "timestamp"]  # no duplicated observation


def validate_series(df: pl.DataFrame) -> pl.DataFrame:
    validated = SeriesSchema.validate(df)

    not_increasing = validated.select(
        (pl.col("timestamp").diff().over("series_id") <= pl.duration(seconds=0)).any()
    ).item()
    if not_increasing:
        raise ValueError("timestamps must be strictly increasing within each series")

    return validated
