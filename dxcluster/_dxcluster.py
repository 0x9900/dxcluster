#!/usr/bin/env python3
#
# BSD 3-Clause License
#
# Copyright (c) 2023 Fred W6BSD
# All rights reserved.
#
# b'DX de SP5NOF:   10136.0  UI5A     FT8 +13dB from KO85 1778Hz   2138Z\r\n'
# b'WWV de W0MU <18Z> :   SFI=93, A=4, K=2, No Storms -> No Storms\r\n'
#
# pylint: disable=no-member,unspecified-encoding

import logging
import os
import random
import re
import signal
import sys
import time

from collections import defaultdict
from dataclasses import astuple, dataclass, field
from datetime import datetime
from itertools import cycle
from queue import Queue
from telnetlib import Telnet
from threading import Event, Thread
from threading import enumerate as thread_enum
from typing import Dict, Any

import sqlite3

from dxcluster import __version__
from dxcluster import adapters
from dxcluster.DXEntity import DXCC
from dxcluster.config import Config

TELNET_RETRY = 3
TELNET_TIMEOUT = 27
FIELDS = ['de', 'frequency', 'dx', 'message', 't_sig', 'de_cont',
          'to_cont', 'de_ituzone', 'to_ituzone', 'de_cqzone',
          'to_cqzone', 'mode', 'signal', 'band', 'time']

SQL_TABLE = """
PRAGMA synchronous = EXTRA;
PRAGMA journal_mode = WAL;
CREATE TABLE IF NOT EXISTS dxspot (
  de TEXT,
  frequency NUMERIC,
  dx TEXT,
  message TEXT,
  t_sig, TIMESTAMP,
  de_cont TEXT,
  to_cont TEXT,
  de_ituzone INTEGER,
  to_ituzone INTEGER,
  de_cqzone INTEGER,
  to_cqzone INTEGER,
  mode TEXT,
  signal INTEGER,
  band INTEGER,
  time TIMESTAMP
);
CREATE UNIQUE INDEX IF NOT EXISTS dxspot_idx_unique on dxspot (de, dx, t_sig);
CREATE INDEX IF NOT EXISTS dxspot_idx_time on dxspot (time DESC);
CREATE INDEX IF NOT EXISTS dxspot_idx_de_cont on dxspot (de_cont);
CREATE INDEX IF NOT EXISTS dxspot_idx_de_cqzone on dxspot (de_cqzone);

CREATE TABLE IF NOT EXISTS wwv (
  SFI INTEGER,
  A INTEGER,
  K INTEGER,
  conditions TEXT,
  time TIMESTAMP
);
CREATE INDEX IF NOT EXISTS wwv_idx_time on wwv (time DESC);
CREATE TABLE IF NOT EXISTS messages (
  de TEXT,
  time TEXT,
  message TEXT,
  timestamp TIMESTAMP
);
CREATE UNIQUE INDEX IF NOT EXISTS messages_idx_unique ON messages (de, time);
"""

DETECT_TYPES = sqlite3.PARSE_DECLTYPES

QUERIES = {}
QUERIES['dxspot'] = f"""
  INSERT OR IGNORE INTO dxspot ({', '.join(f for f in FIELDS)})
  VALUES ({','.join('?' for _ in FIELDS)})
"""
QUERIES['wwv'] = """
  INSERT INTO wwv (SFI, A, K, conditions, time) VALUES (?, ?, ?, ?, ?)
"""
QUERIES['messages'] = """
  INSERT OR IGNORE INTO messages (de, time, message, timestamp) VALUES (?, ?, ?, ?)
"""

if sys.platform == 'linux':
  SIGINFO = signal.SIGUSR1
else:
  SIGINFO = signal.SIGINFO

if os.isatty(sys.stdout.fileno()):
  LOG_FORMAT = '%(asctime)s - %(threadName)s %(lineno)d %(levelname)s - %(message)s'
else:
  LOG_FORMAT = '%(threadName)s %(lineno)d %(levelname)s - %(message)s'

logging.basicConfig(format=LOG_FORMAT, datefmt='%x %X', level=logging.INFO)
LOG = logging.getLogger('dxcluster')


def connect_db(db_name: str, timeout: int=5) -> sqlite3.Connection:
  try:
    conn = sqlite3.connect(db_name, timeout=timeout,
                           detect_types=DETECT_TYPES, isolation_level=None)
    LOG.info("Database: %s", db_name)
  except sqlite3.OperationalError as err:
    LOG.error("Database: %s - %s", db_name, err)
    sys.exit(os.EX_IOERR)
  return conn

def create_db(db_name: str) -> None:
  with connect_db(db_name) as conn:
    curs = conn.cursor()
    curs.executescript(SQL_TABLE)


def get_band(freq: int) -> float:
  # Quick and dirty way to convert frequencies to bands.
  # I should probably have a band plan for each ITU zones.
  # Sorted by the most popular to the least popular band
  _bands = [
    (14000, 14350, 20),
    (7000, 7300, 40),
    (10100, 10150, 30),
    (3500, 4000, 80),
    (21000, 21450, 15),
    (18068, 18168, 17),
    (28000, 29700, 10),
    (50000, 54000, 6),
    (24890, 24990, 12),
    (1800, 2000, 160),
    (144000, 148000, 2),
    (69900, 70500, 4),
    (5258, 5450, 60),
    (420000, 450000, 0.70),
    (219000, 225000, 1.25),
    (1240000, 1300000, 0.23),
    (10000000, 10500000, 0.02),
    (472, 479, 630),
  ]

  for _min, _max, band in _bands:
    if _min <= freq <= _max:
      return band
  LOG.warning("No band for the frequency %s", freq)
  return 0


def dxspider_options(telnet: Telnet, email: str) -> None:
  commands = (
    b'set/wwv on\n',
    b'set/wwv output on\n',
    f'set/email {email}\n'.encode(),
    b'set/dx filter\n',
  )
  for cmd in commands:
    telnet.write(cmd)
    LOG.debug('%s - Command: %s', telnet.host, cmd.decode('UTF-8').rstrip())
    time.sleep(.25)


def ar_options(telnet: Telnet, _: str) -> None:
  commands = (
    b'set/wwv/output on\n',
    b'set/dx/filter\n',
  )
  for cmd in commands:
    telnet.write(cmd)
    LOG.debug('%s - Command: %s', telnet.host, cmd.decode('UTF-8').rstrip())
    time.sleep(.25)


def cc_options(telnet: Telnet, _: str) -> None:
  commands = (b'SET/WWV\n', b'SET/FT4\n', b'SET/FT8\n',  b'SET/PSK\n', b'SET/RTTY\n',
              b'SET/SKIMMER\n')
  for cmd in commands:
    telnet.write(cmd)
    LOG.debug('%s - Command: %s', telnet.host, cmd.decode('UTF-8').rstrip())
    time.sleep(.25)


def login(telnet: Telnet, call: str,  email: str, timeout: int) -> None:
  clusters = {
    "running cc cluster": cc_options,
    "ar-cluster": ar_options,
    "running dxspider": dxspider_options,
  }
  re_spider = re.compile(rf'({"|".join(clusters.keys())})', re.IGNORECASE)
  buffer = []
  expect_exp = [b'your call:', b'login:']

  code, _, match = telnet.expect(expect_exp, timeout)
  if code == -1:
    raise OSError('No login prompt found') from None
  buffer.append(match)
  telnet.write(str.encode(f'{call}\n'))
  try:
    for _ in range(5):
      buffer.append(telnet.read_very_eager())
      time.sleep(0.25)
  except EOFError as err:
    raise OSError(f'{err}: {buffer}') from err

  s_buffer = b'\n'.join(buffer).decode('UTF-8', 'replace')

  if 'invalid callsign' in s_buffer:
    raise OSError('invalid callsign')


  if not (_match := re_spider.search(str(s_buffer))):
    raise OSError('Unknown cluster type')
  match_str = _match.group().lower()
  try:
    LOG.info('%s:%d running %s', telnet.host, telnet.port, match_str)
    set_options = clusters[match_str]
  except KeyError as exp:
    raise OSError('Unknown cluster type') from exp
  set_options(telnet, email)


@dataclass(frozen=True)
class DXSpotRecord:
  # pylint: disable=invalid-name, too-many-instance-attributes
  de: str
  frequency: float
  dx: str
  message: str
  t_sig: datetime
  de_cont: str
  to_cont: str
  de_ituzone: int
  to_ituzone: int
  de_cqzone: int
  to_cqzone: int
  mode: str
  signal: int
  band: int | None = None
  time: datetime | None = None

  def __post_init__(self):
    if self.band is None:
      object.__setattr__(self, 'band', get_band(self.frequency))
    if self.time is None:
      object.__setattr__(self, 'time', datetime.utcnow())


class Static:
  # pylint: disable=too-few-public-methods
  dxcc = DXCC()
  spot_splitter = re.compile(r'[:\s]+').split
  msgparse = re.compile(
      r'^(?P<mode>FT[48]|CW|RTTY|PSK[\d]*)\s+(?P<db>[+-]?\ ?\d+).*'
    ).match


def parse_spot(line: str) -> DXSpotRecord | None:
  #
  # DX de DO4DXA-#:  14025.0  GB22GE       CW 10 dB 25 WPM CQ             1516Z
  # 0.........1.........2.........3.........4.........5.........6.........7.........8
  #           0         0         0         0         0         0         0         0
  if not line:
    return None

  fields: Dict[str, Any] = {}
  try:
    elem = Static.spot_splitter(line)[2:]
    fields.update({
      'de': elem[0].strip('-#'),
      'frequency': float(elem[1]),
      'dx': elem[2],
      'message': ' '.join(elem[3:len(elem) - 1]),
    })
  except ValueError as err:
    LOG.warning("%s | %s", err, re.sub(r'[\n\r\t]+', ' ', line))
    return None

  for c_code in fields['de'].split('/', 1):
    try:
      call_de = Static.dxcc.lookup(c_code)
      break
    except KeyError:
      pass
  else:
    LOG.warning("%s Not found | %s", fields['de'], line)
    return None

  for c_code in fields['dx'].split('/', 1):
    try:
      call_to = Static.dxcc.lookup(c_code)
      break
    except KeyError:
      pass
  else:
    LOG.warning("%s Not found | %s", fields['dx'], line)
    return None

  if (match := Static.msgparse(fields['message'])):
    mode = match.group('mode')
    db_signal = match.group('db')
  else:
    mode = db_signal = None

  t_sig = elem[-1]
  now = datetime.utcnow()
  try:
    t_sig = now.replace(hour=int(t_sig[:2]), minute=int(t_sig[2:4]), second=0, microsecond=0)
  except ValueError:
    t_sig = now.replace(minute=0, second=0, microsecond=0)

  fields['t_sig'] = t_sig
  fields['de_cont'] =  call_de.continent
  fields['to_cont'] =  call_to.continent
  fields['de_ituzone'] =  call_de.ituzone
  fields['to_ituzone'] =  call_to.ituzone
  fields['de_cqzone'] =  call_de.cqzone
  fields['to_cqzone'] =  call_to.cqzone
  fields['mode'] =  mode
  fields['signal'] =  db_signal

  return DXSpotRecord(**fields)


@dataclass(frozen=True)
class WWVRecord:
  # pylint: disable=invalid-name
  sfi: int
  a: int
  k: int
  conditions: str
  time: datetime = field(default=datetime.utcnow(), init=False)


def parse_wwv(line: str) -> WWVRecord | None:
  decoder = re.compile(
    r'.*\sSFI=(?P<SFI>\d+), A=(?P<A>\d+), K=(?P<K>\d+), (?P<conditions>.*)$'
  )

  if not (_match := decoder.match(line)):
    return None
  match = _match.groupdict()
  fields = (
    int(match['SFI']),
    int(match['A']),
    int(match['K']),
    match['conditions'],
  )
  return WWVRecord(*fields)


@dataclass(frozen=True)
class MessageRecord:
  # pylint: disable=invalid-name
  de: str
  time: str
  message: str
  timestamp: datetime = field(default=datetime.utcnow(), init=False)


def parse_message(line: str) -> MessageRecord | None:
  decoder = re.compile(
    r'To ALL de ([-\w]+) \<(\d+Z)>.* : (.*)'
  )

  if not (match := decoder.match(line)):
    return None
  fields = match.groups()
  return MessageRecord(*fields)


class ReString(str):
  def __eq__(self, pattern: object) -> bool:
    if isinstance(pattern, ReString):
      return NotImplemented
    return bool(re.match(str(pattern), self))


class Cluster(Thread):
  def __init__(self, host: str, port: int, queue: Queue, call: str, email: str) -> None:
    super().__init__()
    self.host = host
    self.port = port
    self.queue = queue
    self.call = call
    self.email = email
    self._stop = Event()
    self.timeout = TELNET_TIMEOUT

  def stop(self) -> None:
    self._stop.set()

  def process_spot(self, line: str) -> None:
    try:
      if (record := parse_spot(line)):
        self.queue.put(['dxspot', record])
    except Exception as err:
      LOG.exception("process_spot: you need to deal with this error: %s", err)

  def process_wwv(self, line: str) -> None:
    try:
      if (record := parse_wwv(line)):
        self.queue.put(['wwv', record])
    except Exception as err:
      LOG.exception("wwv: you need to deal with this error: %s", err)

  def process_message(self, line: str) -> None:
    try:
      if (record := parse_message(line)):
        self.queue.put(['messages', record])
    except Exception as err:
      LOG.exception("message: you need to deal with this error: %s", err)

  def run(self) -> None:
    LOG.info("Server: %s:%d", self.host, self.port)
    try:
      with Telnet(self.host, self.port, timeout=self.timeout) as telnet:
        login(telnet, self.call, self.email, self.timeout)
        LOG.info("Sucessful login into %s:%d", self.host, self.port)

        retry = TELNET_RETRY
        while not self._stop.is_set() and retry:
          try:
            _line = telnet.read_until(b'\n', self.timeout)
            line = ReString(_line.decode('UTF-8', 'replace').rstrip())
            Cluster.dumpline(line)
          except EOFError:
            break
          if line == r'^DX de':
            self.process_spot(line)
          elif line == r'^WWV de':
            self.process_wwv(line)
          elif line == r'^To ALL de':
            self.process_message(line)
          elif line == r'^WCY de':
            pass		# this case will be handled soon
          elif line == r'^$':
            retry -= 1
            LOG.warning('Nothing read from: %s retry: %d', self.host, 1 + TELNET_RETRY - retry)
            continue
          else:
            LOG.debug("retry: %d, line %s", retry, line)
          retry = TELNET_RETRY
    except (EOFError, OSError, TimeoutError, UnboundLocalError) as err:
      LOG.error(err)
      return
    LOG.info('Thread finished closing telnet')
    telnet.close()

  @staticmethod
  def dumpline(line: str) -> None:
    if not line:
      return
    with open('/tmp/dxcluster-dump.txt', 'a', encoding='utf-8') as dfd:
      dfd.write(line + "\n")


class SaveRecords(Thread):
  def __init__(self, queue: Queue, db_name: str) -> None:
    super().__init__()
    self.db_name = db_name
    self.queue = queue
    self._stop = Event()

  def stop(self) -> None:
    self._stop.set()

  def running(self) -> bool:
    return not self._stop.is_set()

  def write(self, conn: sqlite3.Connection, table: str, records: list[tuple]) -> None:
    command = QUERIES[table]
    with conn:
      cursor = conn.cursor()
      while True:
        try:
          cursor.executemany(command, records)
          LOG.debug("Table: %s, Data: %3d, row count: %3d: duplicates: %d",
                    table, len(records), cursor.rowcount, len(records) - cursor.rowcount)
          break
        except sqlite3.OperationalError as err:
          LOG.warning("Write error: %s, table: %s,  Queue len: %d",
                      err, table, self.queue.qsize())
          time.sleep(1)

  def run(self):
    with connect_db(self.db_name) as conn:
      while self.running():
        data = defaultdict(list)
        while self.queue.qsize():
          table, record = self.queue.get()
          data[table].append(astuple(record))
        if not data:
          time.sleep(0.5)
          continue
        for table, records in data.items():
          try:
            self.write(conn, table, records)
          except Exception as err:
            LOG.exception('Critical error %s', err)

    LOG.error("SaveRecord thread stopped")

def main():
  config = Config()
  queue = Queue(config.queue_size)
  servers = config.servers
  random.shuffle(servers)
  next_server = cycle(servers).__next__
  adapters.install_adapters()
  create_db(config.db_name)
  log_levels = cycle([logging.DEBUG, logging.INFO])
  LOG.info('Starting dxcluster version: %s', __version__)

  def _sig_handler(_signum, _frame):
    if _signum == signal.SIGHUP:
      LOG.setLevel(next(log_levels))
      LOG.warning('SIGHUP received, switching to %s', logging.getLevelName(LOG.level))
    elif _signum == SIGINFO:
      thread_list = [t for t in thread_enum() if isinstance(t, Cluster)]
      LOG.info('Clusters: %s', ', '.join(t.name for t in thread_list))
      try:
        cache_info = Static.dxcc.get_prefix.cache_info() # ugly but it works.
        rate = 100 * cache_info.hits / (cache_info.misses + cache_info.hits)
        LOG.info("DXEntities cache %s -> %.2f%%", cache_info, rate)
      except AttributeError:
        LOG.info("The cache hasn't been initialized yet")
    elif _signum == signal.SIGINT:
      LOG.critical('Signal ^C received')
      s_thread.stop()
    elif _signum in (signal.SIGQUIT, signal.SIGTERM):
      LOG.info('Quitting')
      s_thread.stop()

  s_thread = SaveRecords(queue, config.db_name)
  s_thread.name = 'SaveRecords'
  s_thread.daemon = True
  s_thread.start()

  LOG.info('Installing signal handlers')
  for sig in (signal.SIGHUP, SIGINFO, signal.SIGINT, signal.SIGQUIT, signal.SIGTERM):
    signal.signal(sig, _sig_handler)

  # Monitor the running threads.
  # Restart reconnect to a different cluster if a cluster thread dies.
  while s_thread.running():
    thread_list = [t for t in thread_enum() if isinstance(t, Cluster)]
    if len(thread_list) >= config.nb_threads:
      time.sleep(1)
      continue
    name, host, port = next_server()
    th_cluster = Cluster(host, port, queue, config.call, config.email)
    th_cluster.name = name
    th_cluster.daemon = True
    th_cluster.timeout = config.telnet_timeout
    th_cluster.start()

  # stopping all the telnet threads
  for cth in (c for c in thread_enum() if isinstance(c, Cluster)):
    LOG.info('Stopping thread: %s', cth.name)
    cth.stop()
  s_thread.stop()


if __name__ == "__main__":
  main()
