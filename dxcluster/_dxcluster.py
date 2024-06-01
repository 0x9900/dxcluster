#!/usr/bin/env python3
#
# Copyright (c) 2023, Fred Cirera
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# b'WWV de W0MU <18Z> :   SFI=93, A=4, K=2, No Storms -> No Storms\r\n'
#
# pylint: disable=unsupported-binary-operation,too-many-arguments

import logging
import os
import random
import re
import signal
import sqlite3
import sys
import time
import typing as t
from collections import defaultdict
from dataclasses import astuple, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import partial, wraps
from itertools import cycle
from queue import Full as QFull
from queue import Queue
from telnetlib import Telnet
from threading import Event, Thread
from threading import enumerate as thread_enum

from DXEntity import DXCC, DXCCRecord

from .adapters import install_adapters
from .config import Config

__version__ = "0.0.5"

TELNET_TIMEOUT = 27
# The maximum amount of time we stay connected to one server.
TELNET_MAX_TIME = 3600 * 4


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
CREATE UNIQUE INDEX IF NOT EXISTS wwv_idx_time_unique on wwv (time DESC);

CREATE TABLE IF NOT EXISTS messages (
  de TEXT,
  time TEXT,
  message TEXT,
  timestamp TIMESTAMP
);
CREATE UNIQUE INDEX IF NOT EXISTS messages_idx_unique ON messages (de, time);
"""

DETECT_TYPES = sqlite3.PARSE_DECLTYPES


class Tables(Enum):
  DXSPOT = 1
  WWV = 2
  MESSAGE = 3


QUERIES = {
  Tables.DXSPOT: """
  INSERT OR IGNORE INTO dxspot
    (de, frequency, dx, message, t_sig, de_cont, to_cont, de_ituzone,
    to_ituzone, de_cqzone, to_cqzone, mode, signal, band, time)
  VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  """,
  Tables.WWV: """
  INSERT OR IGNORE  INTO wwv (SFI, A, K, conditions, time)
  VALUES (?, ?, ?, ?, ?)
  """,
  Tables.MESSAGE: """
  INSERT OR IGNORE INTO messages (de, time, message, timestamp)
  VALUES (?, ?, ?, ?)
  """,
}


if sys.platform == 'linux':
  SIGINFO = signal.SIGUSR1		# pylint: disable=no-member
else:
  SIGINFO = signal.SIGINFO		# pylint: disable=no-member

if os.isatty(sys.stdout.fileno()):
  LOG_FORMAT = '%(asctime)s - %(threadName)s %(lineno)d %(levelname)s - %(message)s'
else:
  LOG_FORMAT = '%(threadName)s %(lineno)d %(levelname)s - %(message)s'

logging.basicConfig(format=LOG_FORMAT, datefmt='%x %X', level=logging.INFO)
LOG = logging.getLogger('dxcluster')


def connect_db(db_name: str, timeout: int = 5) -> sqlite3.Connection:
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
  commands = (b'SET/WWV\n', b'SET/FT4\n', b'SET/FT8\n', b'SET/PSK\n', b'SET/RTTY\n',
              b'SET/SKIMMER\n')
  for cmd in commands:
    telnet.write(cmd)
    LOG.debug('%s - Command: %s', telnet.host, cmd.decode('UTF-8').rstrip())
    time.sleep(.25)


def login(telnet: Telnet, call: str, email: str, timeout: int) -> None:
  # pylint: disable=too-many-locals
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
    LOG.info('%s:%d running %s', telnet.host, telnet.port, match_str)  # type: ignore
    set_options = clusters[match_str]
  except KeyError as exp:
    raise OSError('Unknown cluster type') from exp
  set_options(telnet, email)


@dataclass(frozen=True, slots=True)
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
      object.__setattr__(self, 'time', datetime.now(timezone.utc))


class Static:
  # pylint: disable=too-few-public-methods
  dxcc = DXCCRecord
  spot_splitter = partial(re.compile(r'[:\s]+').split, maxsplit=5)
  msgparse = re.compile(
    r'(?P<mode>FT[48]|CW|RTTY|PSK[\d]*)\s+(?P<db>[+-]?\ ?\d+).*\s((?P<t_sig>\d{4}Z)|).*'
  ).match


def parse_spot(line: str) -> DXSpotRecord | None:
  #
  # DX de DO4DXA-#:  14025.0  GB22GE       CW 10 dB 25 WPM CQ             1516Z
  # 0.........1.........2.........3.........4.........5.........6.........7.........8
  #           0         0         0         0         0         0         0         0
  if not line:
    return None

  fields: t.Dict[str, t.Any] = {}
  try:
    elem = Static.spot_splitter(line)[2:]
    fields.update({
      'de': elem[0].strip('-#'),
      'frequency': float(elem[1]),
      'dx': elem[2],
      'message': elem[3],
    })
  except (IndexError, ValueError) as err:
    LOG.warning("%s | %s", err, re.sub(r'[\n\r\t]+', ' ', line))
    return None

  for c_code in fields['de'].split('/', 1):
    try:
      call_de = Static.dxcc.lookup(c_code)  # type: ignore # pylint: disable=no-member
      break
    except KeyError:
      pass
  else:
    LOG.debug("%s Not found | %s", fields['de'], line)
    return None

  for c_code in fields['dx'].split('/', 1):
    try:
      call_to = Static.dxcc.lookup(c_code)  # type: ignore # pylint: disable=no-member
      break
    except KeyError:
      pass
  else:
    LOG.debug("%s Not found | %s", fields['dx'], line)
    return None

  match = Static.msgparse(fields['message'])
  if match:
    mode = match.group('mode')
    db_signal = match.group('db')
    db_signal = int(db_signal.replace(' ', ''))
  else:
    mode = db_signal = None

  now = datetime.now(timezone.utc)
  if match and match.group('t_sig'):
    t_sig = match.group('t_sig')
    t_sig = now.replace(hour=int(t_sig[:2]), minute=int(t_sig[2:4]), second=0, microsecond=0)
  else:
    t_sig = now.replace(minute=0, second=0, microsecond=0)

  fields['t_sig'] = t_sig
  fields['de_cont'] = call_de.continent
  fields['to_cont'] = call_to.continent
  fields['de_ituzone'] = int(call_de.ituzone)
  fields['to_ituzone'] = int(call_to.ituzone)
  fields['de_cqzone'] = int(call_de.cqzone)
  fields['to_cqzone'] = int(call_to.cqzone)
  fields['mode'] = mode
  fields['signal'] = db_signal

  return DXSpotRecord(**fields)


@dataclass(frozen=True, slots=True)
class WWVRecord:
  # pylint: disable=invalid-name
  sfi: int
  a: int
  k: int
  conditions: str
  time: datetime | None = None

  def __post_init__(self):
    if self.time is None:
      _curtime = datetime.now(timezone.utc).replace(microsecond=0)
      object.__setattr__(self, 'time', _curtime)


def parse_wwv(line: str) -> WWVRecord | None:
  decoder = re.compile(
    r'.*\sSFI=(?P<SFI>\d+), A=(?P<A>\d+), K=(?P<K>\d+), ((?P<c1>.*)(\s|)->(\s|)(?P<c2>.*)|)'
  )

  if not (_match := decoder.match(line)):
    LOG.warning('WWV parse error: %s', line)
    return None
  match = _match.groupdict()
  conditions = []
  for key in ('c1', 'c2'):
    if match[key] and not match[key].startswith('No Storm'):
      conditions.append(match[key].rstrip())
  s_cond = ' - '.join(conditions)
  return WWVRecord(int(match['SFI']), int(match['A']), int(match['K']), s_cond)


def parse_wcy(line: str) -> WWVRecord | None:
  # WCY de DK0WCY-2 <01> : K=2 expK=0 A=7 R=85 SFI=134 SA=eru GMF=qui Au=no
  decoder = re.compile(
    r'WCY de .*\sK=(?P<K>\d+).*\sA=(?P<A>\d+).*\sSFI=(?P<SFI>\d+).*'
  )
  if not (_match := decoder.match(line)):
    LOG.warning('WCY parse error: %s', line)
    return None
  match = _match.groupdict()
  return WWVRecord(int(match['SFI']), int(match['A']), int(match['K']), '')


@dataclass(frozen=True, slots=True)
class MessageRecord:
  # pylint: disable=invalid-name
  de: str
  time: str
  message: str
  timestamp: datetime | None = field(init=False, default=None)

  def __post_init__(self):
    if self.timestamp is None:
      _now = datetime.now(timezone.utc).replace(microsecond=0)
      object.__setattr__(self, 'timestamp', _now)


def parse_message(line: str) -> MessageRecord | None:
  decoder = re.compile(
    r'To ALL de ([-\w]+) <(\d{4}Z)>.* : (.*)'
  )

  if not (match := decoder.match(line)):
    return None
  fields = tuple(match.groups())
  return MessageRecord(*fields)


def block_exceptions(func: t.Callable[..., t.Any]) -> t.Any:
  @wraps(func)
  def wrapper(*args: t.Any, **kwargs: t.Any) -> t.Any:
    try:
      return func(*args, **kwargs)
    except Exception as err:
      with open('/tmp/dxcluster-dump.txt', 'a', encoding='utf-8') as dfd:
        dfd.write(args[1] + "\n")
      LOG.exception("process_spot: you need to deal with this error: %s", err)
    return None
  return wrapper


class ReString(str):
  def __eq__(self, pattern: object) -> bool:
    if isinstance(pattern, ReString):
      return NotImplemented
    return bool(re.match(str(pattern), self))


class Cluster(Thread):
  # pylint: disable=too-many-instance-attributes
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

  @block_exceptions
  def process_spot(self, line: str) -> None:
    if (record := parse_spot(line)):
      try:
        self.queue.put((Tables.DXSPOT, record))
      except QFull:
        pass

  @block_exceptions
  def process_wwv(self, line: str) -> None:
    if (record := parse_wwv(line)):
      try:
        self.queue.put((Tables.WWV, record))
      except QFull:
        pass

  @block_exceptions
  def process_wcy(self, line: str) -> None:
    if (record := parse_wcy(line)):
      try:
        self.queue.put((Tables.WWV, record))
      except QFull:
        pass

  @block_exceptions
  def process_message(self, line: str) -> None:
    if (record := parse_message(line)):
      try:
        self.queue.put((Tables.MESSAGE, record))
      except QFull:
        pass

  def run(self) -> None:
    LOG.info("Server: %s:%d", self.host, self.port)
    try:
      with Telnet(self.host, self.port, timeout=self.timeout) as telnet:
        login(telnet, self.call, self.email, self.timeout)
        LOG.info("Sucessful login into %s:%d", self.host, self.port)

        # Add a random time to avoid having all the servers disconnect simultaneously.
        timer = Timer(TELNET_MAX_TIME + random.randint(-1800, 1800))
        while not self._stop.is_set() and next(timer):
          try:
            _line = telnet.read_until(b'\n', self.timeout)
            line = ReString(_line.decode('UTF-8', 'replace').rstrip())
          except EOFError:
            break
          if line == r'^$':
            LOG.warning('Nothing read from: %s', self.host)
            timer.reduce()
          elif line == r'^DX de':
            self.process_spot(line)
          elif line == r'^WWV de':
            self.process_wwv(line)
          elif line == r'^To ALL de':
            self.process_message(line)
          elif line == r'^WCY de':
            self.process_wcy(line)

    except (EOFError, OSError, TimeoutError, UnboundLocalError) as err:
      LOG.error("%s - Error: %s", self.name, err)
    except StopIteration:
      LOG.info('Thread finished closing telnet to: %s', self.host)
      telnet.close()


class Timer:
  def __init__(self, max_time):
    self.max_time = self.current_time + max_time

  def __iter__(self):
    return self

  def reduce(self):
    self.max_time /= 2

  def __next__(self):
    if self.current_time > self.max_time:
      raise StopIteration
    return True

  @property
  def current_time(self):
    return time.time()


class QueueIterator:
  def __init__(self, queue):
    self.queue = queue

  def __iter__(self):
    return self

  def __next__(self):
    if self.queue.empty():
      raise StopIteration
    return self.queue.get()


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

  def read_queue(self):
    while self.queue.qsize() == 0:
      time.sleep(1)
    data = defaultdict(list)
    q_iterator = QueueIterator(self.queue)
    for count, (table, record) in enumerate(q_iterator, start=1):
      data[table].append(astuple(record))
      if count >= 512:
        break
    return data

  def write(self, conn: sqlite3.Connection, table: Tables, records: list[tuple]) -> None:
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
          LOG.warning("Write error: %s, table: %s, Queue len: %4d/%d",
                      err, table, self.queue.qsize(), self.queue.maxsize)
          time.sleep(1)

  def run(self):
    with connect_db(self.db_name) as conn:
      while self.running():
        data = self.read_queue()
        for table, records in data.items():
          try:
            self.write(conn, table, records)
          except Exception as err:
            LOG.exception('Critical error %s', err)

    LOG.warning("SaveRecord thread stopped")


def make_queue(config):
  if isinstance(config.queue_size, str) and config.queue_size.lower() == 'auto':
    _qsize = 512 * 6
  else:
    _qsize = config.queue_size

  qsize = int(_qsize * config.nb_threads)
  return Queue(qsize)


def main():
  Static.dxcc = DXCC(cache_size=8192)
  config = Config()
  queue = make_queue(config)
  servers = config.servers
  random.shuffle(servers)
  next_server = cycle(servers).__next__
  install_adapters()
  create_db(config.db_name)
  log_levels = cycle([logging.DEBUG, logging.INFO])
  LOG.info('Starting dxcluster version: %s', __version__)

  def _sig_handler(_signum, _frame):
    if _signum == signal.SIGHUP:
      LOG.setLevel(next(log_levels))
      LOG.warning('SIGHUP received, switching to %s', logging.getLevelName(LOG.level))
    elif _signum == SIGINFO:
      thread_list = [t for t in thread_enum() if isinstance(t, Cluster)]
      LOG.info('Running dxcluster version: %s', __version__)
      LOG.info('Clusters: %s', ', '.join(t.name for t in thread_list))
      try:
        cache_info = Static.dxcc.cache_info()  # ugly but it works.
        rate = 100 * cache_info.hits / (cache_info.misses + cache_info.hits)
        LOG.info("DXEntities cache %s -> %.2f%%", cache_info, rate)
      except (AttributeError, ZeroDivisionError):
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
      time.sleep(3)
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
