# Copyright 2025 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PyGrain data sources implementation."""

from __future__ import annotations

import dataclasses
import functools
from typing import Any, SupportsIndex

from etils import epath
from etils import epy
from grain import python as grain
from kauldron import kd

with epy.lazy_imports():
    # pylint: disable=g-import-not-at-top   # pytype: disable=import-error
  import pyarrow as pa
  import pyarrow.parquet as pq
    # pylint: enable=g-import-not-at-top   # pytype: disable=import-error

# TODO(epot): Move to kd.data.py (or `kd.contrib` ?)


@dataclasses.dataclass(frozen=True, kw_only=True)
class ParquetDataSource(grain.RandomAccessDataSource):
  """A data source that reads from a Parquet file."""

  path: epath.PathLike | list[epath.PathLike]

  @functools.cached_property
  def table(self) -> pa.Table:
    """Parquet table."""
    # TODO(epot): Multi-path support (currently raises an error when passing
    # a list of  `epath.Path().open('rb')`)

    # Normalize path to a list.
    if isinstance(self.path, epath.PathLikeCls):
      with epath.Path(self.path).open(mode='rb') as f:
        return pq.read_table(f)
    else:
      return pq.read_table(self.path)

  def __len__(self) -> int:
    return len(self.table)

  def __getitem__(self, record_key: SupportsIndex) -> dict[str, Any]:
    record = self.table.take([record_key])
    return {k: v[0] for k, v in record.to_pydict().items()}


@dataclasses.dataclass(frozen=True, kw_only=True)
class Parquet(kd.data.py.DataSourceBase):

  path: epath.PathLike | list[epath.PathLike]

  @functools.cached_property
  def data_source(self) -> grain.RandomAccessDataSource:
    return ParquetDataSource(path=self.path)
