#
# Copyright (c) 2023, Fred Cirera
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
#

"""Save datetime object as timestamp in SQLite"""

import sqlite3

from datetime import datetime

def adapt_datetime(t_stamp):
  return t_stamp.timestamp()

def convert_datetime(t_stamp):
  try:
    return datetime.fromtimestamp(float(t_stamp))
  except ValueError:
    return None

def install_adapters():
  sqlite3.register_adapter(datetime, adapt_datetime)
  sqlite3.register_converter('timestamp', convert_datetime)
