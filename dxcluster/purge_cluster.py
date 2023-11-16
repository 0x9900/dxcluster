#!/usr/bin/env python3
#
# BSD 3-Clause License
#
# Copyright (c) 2023 Fred W6BSD
# All rights reserved.
#
#

import logging
import sqlite3

from datetime import datetime, timedelta

from dxcluster import adapters
from dxcluster.config import Config

logging.basicConfig(
  format='%(asctime)s - %(lineno)d %(levelname)s - %(message)s',
  level=logging.INFO
)

def purge(conn, purge_time):
  logging.info("Purge entries from before: %s", purge_time.isoformat())
  curs = conn.cursor()
  try:
    curs.execute('BEGIN TRANSACTION')
    curs.execute('DELETE FROM dxspot WHERE time < ?;', (purge_time,))
    logging.info('%d record deleted', curs.rowcount)
    curs.execute("COMMIT")
  except conn.error:
    curs.execute("ROLLBACK")


def main():
  adapters.install_adapters()
  config = Config()
  delta = timedelta(hours=config.purge_time)
  purge_time = datetime.utcnow() - delta

  logging.info('Database: %s, timeout %d', config.db_name, config.db_timeout)
  logging.info('Delta: %0.1f days', delta.days)
  conn = sqlite3.connect(
    config.db_name, timeout=5, detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES
  )
  conn.isolation_level = None

  try:
    purge(conn, purge_time)
  except sqlite3.OperationalError as err:
    logging.error(err)


if __name__ == "__main__":
  main()
