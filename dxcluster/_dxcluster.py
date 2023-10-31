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
import socket
import sys
import time

from collections import defaultdict
from collections import namedtuple
from datetime import datetime
from itertools import cycle
from queue import Queue, Full
from telnetlib import Telnet
from threading import Event, Lock, Thread
from threading import enumerate as thread_enum

import sqlite3

from dxcluster import adapters
from dxcluster.DXEntity import DXCC
from dxcluster.config import Config

TELNET_TIMEOUT = 27
FIELDS = ['DE', 'FREQUENCY', 'DX', 'MESSAGE', 'T_SIG', 'DE_CONT',
          'TO_CONT', 'DE_ITUZONE', 'TO_ITUZONE', 'DE_CQZONE',
          'TO_CQZONE', 'MODE', 'SIGNAL', 'BAND', 'TIME']

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


def connect_db(db_name, timeout=5):
  try:
    conn = sqlite3.connect(db_name, timeout=timeout,
                           detect_types=DETECT_TYPES, isolation_level=None)
    LOG.info("Database: %s", db_name)
  except sqlite3.OperationalError as err:
    LOG.error("Database: %s - %s", db_name, err)
    sys.exit(os.EX_IOERR)
  return conn

def create_db(db_name):
  with connect_db(db_name) as conn:
    curs = conn.cursor()
    curs.executescript(SQL_TABLE)


def get_band(freq):
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


def dxspider_options(telnet, email):
  commands = (
    b'set/dx filter\n',
    b'set/wwv on\n',
    b'set/wwv output on\n',
    f'set/email {email}\n'.encode(),
  )
  for cmd in commands:
    telnet.write(cmd)
    LOG.debug('%s - Command: %s', telnet.host, cmd.decode('UTF-8').rstrip())
    time.sleep(.25)

def ar_options(telnet, _):
  commands = (
    b'set/dx/filter',
    b'set/wwv/output on',
  )
  for cmd in commands:
    telnet.write(cmd)
    LOG.debug('%s - Command: %s', telnet.host, cmd.decode('UTF-8').rstrip())
    time.sleep(.25)


def cc_options(telnet, _):
  commands = (b'SET/WWV\n', b'SET/FT4\n', b'SET/FT8\n',  b'SET/PSK\n', b'SET/RTTY\n',
              b'SET/SKIMMER\n')
  for cmd in commands:
    telnet.write(cmd)
    LOG.debug('%s - Command: %s', telnet.host, cmd.decode('UTF-8').rstrip())
    time.sleep(.25)

def login(call, telnet, email, timeout):
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
    raise OSError(f'{err}: {buffer}')

  buffer = b'\n'.join(buffer).decode('UTF-8', 'replace')
  if 'invalid callsign' in buffer:
    raise OSError('invalid callsign')

  match = re_spider.search(buffer)
  if not match:
    raise OSError('Unknown cluster type')
  match_str = match.group().lower()
  try:
    LOG.info('%s:%d running %s', telnet.host, telnet.port, match_str)
    set_options = clusters[match_str]
  except KeyError:
    raise OSError('Unknown cluster type')
  set_options(telnet, email)


def parse_spot(line):
  #
  # DX de DO4DXA-#:  14025.0  GB22GE       CW 10 dB 25 WPM CQ             1516Z
  # 0.........1.........2.........3.........4.........5.........6.........7.........8
  #           0         0         0         0         0         0         0         0
  if not hasattr(parse_spot, 'dxcc'):
    parse_spot.dxcc = DXCC()

  if not hasattr(parse_spot, 'splitter'):
    parse_spot.splitter = re.compile(r'[:\s]+').split

  if not hasattr(parse_spot, 'msgparse'):
    parse_spot.msgparse = re.compile(
      r'^(?P<mode>FT[48]|CW|RTTY|PSK[\d]*)\s+(?P<db>[+-]?\ ?\d+).*'
    ).match

  elem = parse_spot.splitter(line)[2:]

  try:
    fields = [
      elem[0].strip('-#'),
      float(elem[1]),
      elem[2],
      ' '.join(elem[3:len(elem) - 1]),
    ]
  except ValueError as err:
    LOG.warning("%s | %s", err, re.sub(r'[\n\r\t]+', ' ', line))
    return None

  for c_code in fields[0].split('/', 1):
    try:
      call_de = parse_spot.dxcc.lookup(c_code)
      break
    except KeyError:
      pass
  else:
    LOG.warning("%s Not found | %s", fields[0], line)
    return None

  for c_code in fields[2].split('/', 1):
    try:
      call_to = parse_spot.dxcc.lookup(c_code)
      break
    except KeyError:
      pass
  else:
    LOG.warning("%s Not found | %s", fields[2], line)
    return None

  match = parse_spot.msgparse(fields[3])
  if match:
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

  fields.extend([
    t_sig,
    call_de.continent,
    call_to.continent,
    call_de.ituzone,
    call_to.ituzone,
    call_de.cqzone,
    call_to.cqzone,
    mode,
    db_signal,
  ])
  return DXSpotRecord(fields)


def parse_wwv(line):
  decoder = re.compile(
    r'.*\sSFI=(?P<SFI>\d+), A=(?P<A>\d+), K=(?P<K>\d+), (?P<conditions>.*)$'
  )
  match = decoder.match(line)
  if not match:
    return None
  mach = match.groupdict()
  fields = [
    int(match['SFI']),
    int(match['A']),
    int(match['K']),
    match['conditions'],
  ]
  return WWVRecord(fields)


def parse_message(line):
  decoder = re.compile(
    r'To ALL de ([-\w]+) \<(\d+Z)>.* : (.*)'
  )
  match = decoder.match(line)
  if not match:
    return None
  fields = match.groups()
  return MessageRecord(fields)


class MessageRecord(namedtuple("MessageRecord", "de, time, message, timestamp")):
  def __new__(cls, items):
    _items = list(items)
    _items.append(datetime.utcnow())
    return tuple.__new__(cls, _items)

  def as_list(self):
    return [self.de, self.time, self.message, self.timestamp]


class WWVRecord(namedtuple("WWVRecord", "SFI, A, K, conditions, time")):
  def __new__(cls, items):
    _items = items
    _items.append(datetime.utcnow())
    return tuple.__new__(cls, _items)

  def as_list(self):
    return [self.SFI, self.A, self.K, self.conditions, self.time]

class DXSpotRecord(namedtuple('DXSpotRecord', FIELDS)):
  def __new__(cls, items):
    _items = items
    _items.append(get_band(_items[1]))
    _items.append(datetime.utcnow())
    return tuple.__new__(cls, _items)

  def as_list(self):
    return [self.DE, self.FREQUENCY, self.DX, self.MESSAGE, self.T_SIG, self.DE_CONT,
            self.TO_CONT, self.DE_ITUZONE, self.TO_ITUZONE, self.DE_CQZONE,
            self.TO_CQZONE, self.MODE, self.SIGNAL, self.BAND, self.TIME]


class Cluster(Thread):
  def __init__(self, host, port, queue, call, email):
    super().__init__()
    self.host = host
    self.port = port
    self.queue = queue
    self.call = call
    self.email = email
    self._stop = Event()
    self._timeout = TELNET_TIMEOUT

  def stop(self):
    self._stop.set()

  def process_spot(self, line):
    try:
      record = parse_spot(line)
      if record:
        self.queue.put(['dxspot', record])
    except Exception as err:
      LOG.exeption("process_spot: you need to deal with this error: %s", err)

  def process_wwv(self, line):
    try:
      record = parse_wwv(line)
      if record:
        self.queue.put(['wwv', record])
    except Exception as err:
      LOG.exeption("wwv: you need to deal with this error: %s", err)

  def process_message(self, line):
    try:
      record = parse_message(line)
      if record:
        self.queue.put(['messages', record])
    except Exception as err:
      LOG.exeption("message: you need to deal with this error: %s", err)

  def run(self):
    trace(self.name, self.host, self.port, 'Start')
    try:
      LOG.info(f"Server: %s:%d", self.host, self.port)
      self.telnet = Telnet(self.host, self.port, timeout=self._timeout)
      login(self.call, self.telnet, self.email, self.timeout)
      LOG.info(f"Sucessful login into %s:%d", self.host, self.port)
    except (EOFError, OSError, TimeoutError, UnboundLocalError) as err:
      LOG.error(err)
      return

    counter = 200000
    while not self._stop.is_set() and counter:
      try:
        line = self.telnet.read_until(b'\n', self.timeout)
        if not line:
          LOG.warning('Nothing read from: %s in %d seconds disconnecting',
                      self.host,  self._timeout)
          break
        line = line.decode('UTF-8', 'replace').rstrip()
        if line.startswith('DX de'):
          self.process_spot(line)
        elif line.startswith('WWV de'):
          self.process_wwv(line)
        elif line.startswith('To ALL de'):
          self.process_message(line)
        elif line.startswith('WCY de '):
          # Not processed yet
          pass
        else:
          LOG.warning("Counter: %d, Line: %s", counter, line)
      except EOFError:
        break

      counter -= 1
      if counter % 10000 == 0:
        trace(self.name, self.host, self.port, f'Counter: {counter}')

    trace(self.name, self.host, self.port, 'Shutdown')
    LOG.info('Thread finished closing telnet')
    self.telnet.close()

    @property
    def timeout(self):
      return self._timeout

    @timeout.setter
    def timeout(self, tm_out):
      self._timeout = tm_out


class SaveRecords(Thread):
  def __init__(self, queue, db_name):
    super().__init__()
    self.db_name = db_name
    self.queue = queue
    self._stop = Event()

  def stop(self):
    self._stop.set()

  def running(self):
    return not self._stop.is_set()

  def write(self, conn, table, records):
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
      cursor = conn.cursor()
      while self.running():
        data = defaultdict(list)
        while self.queue.qsize():
          table, record = self.queue.get()
          data[table].append(record.as_list())

        if not data:
          time.sleep(0.5)
          continue

        for table, records in data.items():
          try:
            self.write(conn, table, records)
          except Exception as err:
            LOG.exeption('Critical error %s', err)

    LOG.error("SaveRecord thread stopped")


def trace(name, host, port, msg):
  if not hasattr(trace, '_lock'):
    trace._lock = Lock()

  trace._lock.acquire()
  now = datetime.now().isoformat()
  with open('/tmp/dxcluster-trace.txt', 'a', encoding='utf-8') as tfd:
    tfd.write(f'{name}, {host}, {port}, {now}, {msg}\n')
  trace._lock.release()


def main():
  config = Config()
  queue = Queue(config.queue_size)
  servers = config.servers
  random.shuffle(servers)
  next_server = cycle(servers).__next__
  adapters.install_adapters()
  create_db(config.db_name)
  log_levels = cycle([10, 20, 30])

  def _sig_handler(_signum, _frame):
    if _signum == signal.SIGHUP:
      LOG.setLevel(log_levels.__next__())
      LOG.warning('SIGHUP received, switching to %s', logging._levelToName[LOG.level])
    elif _signum == SIGINFO:
      cache_info = parse_spot.dxcc.get_prefix.cache_info() # ugly but it works.
      rate = 100 * cache_info.hits / (cache_info.misses + cache_info.hits)
      LOG.info("DXEntities cache %s -> %.2f%%", cache_info, rate)
      thread_list = [t for t in thread_enum() if isinstance(t, Cluster)]
      LOG.info('Clusters: %s', ', '.join(t.name for t in thread_list))
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
  for sig in (signal.SIGHUP, SIGINFO, signal.SIGINT,
              signal.SIGQUIT, signal.SIGTERM):
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
