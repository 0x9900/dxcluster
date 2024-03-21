#
# Copyright (c) 2023, Fred Cirera
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
#

import logging
import sqlite3
import time
from datetime import datetime, timedelta

from .adapters import install_adapters
from .config import Config

logging.basicConfig(
  format='%(asctime)s %(name)s:%(lineno)3d - %(levelname)s - %(message)s', datefmt='%x %X',
  level=logging.INFO
)


class Timer:
  def __init__(self):
    self.elapse = self.start = 0

  def __enter__(self):
    self.start = time.time()
    return self

  def __exit__(self, *args):
    self.elapse = time.time() - self.start


def purge(conn, purge_time):
  logging.info("Purge entries from before: %s", purge_time.isoformat())
  curs = conn.cursor()
  try:
    with Timer() as ptime:
      curs.execute('BEGIN TRANSACTION')
      curs.execute('DELETE FROM dxspot WHERE time < ?;', (purge_time,))
      deleted = curs.rowcount
      curs.execute("COMMIT")
    logging.info('%d record deleted in %f', deleted, ptime.elapse)
  except sqlite3.OperationalError as err:
    logging.error(err)
    curs.execute("ROLLBACK")


def main():
  install_adapters()
  config = Config()
  delta = timedelta(hours=config.purge_time)
  purge_time = datetime.utcnow() - delta

  logging.info('Database: %s, timeout %d', config.db_name, config.db_timeout)
  logging.info('Delta: %0.1f days', delta.days)
  conn = sqlite3.connect(
    config.db_name, timeout=5, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
  )
  conn.isolation_level = None

  try:
    purge(conn, purge_time)
  except sqlite3.OperationalError as err:
    logging.error(err)


if __name__ == "__main__":
  main()
