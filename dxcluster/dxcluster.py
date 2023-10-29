#!/usr/bin/env python3
#
# BSD 3-Clause License
#
# Copyright (c) 2022-2023 Fred W6BSD
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
from threading import Event, Thread
from threading import enumerate as thread_enum

import sqlite3
import yaml

from dxcluster import adapters
from dxcluster.DXEntity import DXCC

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

TELNET_TIMEOUT = 17
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


def spider_options(_, telnet, email):
  commands = (
    (b'set/dx/filter\n', b'DX filter.*\n'),
    (b'set/wwv on\n', b'set/wwv on\n'),
    (b'set/wwv/output on\n', b'WWV output set.*\n'),
    (f'set/station/email {email}\n'.encode(), b'Email address set.*\n'),
  )
  for cmd, reply_ex in commands:
    telnet.write(cmd)
    LOG.info('%s - Command: %s', telnet.host, cmd.decode('UTF-8').rstrip())


def cc_options(call, telnet, _):
  prompt = str.encode(f'{call} de .*\n')
  commands = (b'SET/WWV\n', b'SET/FT4\n', b'SET/FT8\n',  b'SET/PSK\n', b'SET/RTTY\n',
              b'SET/SKIMMER\n')
  for cmd in commands:
    telnet.write(cmd)
    LOG.info('%s - Command: %s', telnet.host, cmd.decode('UTF-8').rstrip())

def login(call, telnet, email):
  expect_exp = [
    b'Running CC Cluster.*\n',
    b'AR-Cluster.*\n',
    b'running DXSpider.*\n',
  ]
  try:
    for _ in range(5):
      code, _,  match = telnet.expect(expect_exp, TELNET_TIMEOUT)
      if code == 0:
        set_options = cc_options
      elif code == 1:
        set_options = spider_options
      elif code == 2:
        raise OSError('DX Spider cluster')
      break
  except socket.timeout:
    raise OSError('Connection timeout') from None
  except EOFError as err:
    raise OSError(err) from None

  expect_exp = [
    b'.*enter your call.*\n',
    b'.*enter your amateur radio callsign.*\n'
  ]
  prompt = [s.encode('utf-8') for s in  (f'{call} de .*\n', 'not a valid callsign')]
  telnet.write(str.encode(f'{call}\n'))
  code, match, b = telnet.expect(prompt, TELNET_TIMEOUT)
  if code == -1:
    raise OSError('Login error, this server looks like a non valid dx cluster')
  elif code == 0:
    match = match.group().decode('UTF-8')
    LOG.info('%s - Reply: %s', telnet.host, match.strip())
    set_options(call, telnet, email)
  elif code == 1:
    LOG.error('Login error %s %s', call, match.group())
    raise OSError('Login error, invalid login')


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
    t_sig = now.replace(minute=0, second=0, micrsecond=0)

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

  def stop(self):
    self._stop.set()

  def process_spot(self, line):
    try:
      record = parse_spot(line)
      if record:
        self.queue.put(['dxspot', record])
    except Exception as err:
      LOG.critical("process_spot: you need to deal with this error: %s", err)

  def process_wwv(self, line):
    try:
      record = parse_wwv(line)
      if record:
        self.queue.put(['wwv', record])
    except Exception as err:
      LOG.critical("wwv: you need to deal with this error: %s", err)

  def process_message(self, line):
    try:
      record = parse_message(line)
      if record:
        self.queue.put(['messages', record])
    except Exception as err:
      LOG.critical("message: you need to deal with this error: %s", err)

  def run(self):
    try:
      LOG.info(f"Server: %s:%d", self.host, self.port)
      self.telnet = Telnet(self.host, self.port, timeout=TELNET_TIMEOUT)
      login(self.call, self.telnet, self.email)
      LOG.info(f"Sucessful login into %s:%d", self.host, self.port)
    except (OSError, TimeoutError, UnboundLocalError) as err:
      LOG.error(err)
      return

    counter = 0
    while not self._stop.is_set():
      try:
        line = self.telnet.read_until(b'\n', TELNET_TIMEOUT)
        if not line:
          LOG.warning('Nothing read from: %s in %d seconds disconnecting',
                      self.host,  TELNET_TIMEOUT)
          break
        counter += 1
        if counter > 200000:
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
          LOG.warning(line)
      except EOFError:
        break
    LOG.info('Thread finished')


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

  @staticmethod
  def write(conn, table, records):
    command = QUERIES[table]
    with conn:
      cursor = conn.cursor()
      while True:
        try:
          cursor.executemany(command, records)
          LOG.debug("Row Count: %d, data len: %d", cursor.rowcount, len(records))
          break
        except sqlite3.OperationalError as err:
          LOG.warning("Write error: %s, Queue size: %s", err, self.queue.qsize())
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
            SaveRecords.write(conn, table, records)
          except Exception as err:
            LOG.critical('Critical error %s', err)


    LOG.error("SaveRecord thread exit")

class Config:
  DIRS = ['.', '~/.local']
  _instance = None
  config_data = None
  def __new__(cls, *args, **kwargs):
    if not cls._instance:
      cls._instance = super(Config, cls).__new__(cls)
      cls._instance.config_data = None
    return cls._instance

  def __init__(self):
    if not self.config_data:
      self.config_data = Config.read_config()

  @staticmethod
  def read_config():
    filename = 'dxcluster.yaml'
    for path in [os.path.expanduser(p) for p in Config.DIRS]:
      full_path = os.path.join(path, filename)
      try:
        with open(full_path, 'r', encoding="utf-8") as cfg:
          return yaml.safe_load(cfg)
      except FileNotFoundError:
        pass
    raise FileNotFoundError('No config file found')

  @property
  def call(self):
    return self.config_data['call'].upper()

  @property
  def db_name(self):
    return self.config_data['db_name']

  @property
  def nb_threads(self):
    return self.config_data.get('nb_threads', 2)

  @property
  def telnet_timeout(self):
    return self.config_data.get('telnet_timeout', 60)

  @property
  def servers(self):
    return self.config_data['servers']

  @property
  def queue_size(self):
    return self.config_data.get('queue_size', 512)

  @property
  def email(self):
    return self.config_data.get('email', f'{self.call}@arrl.org')


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
      LOG.setLevel(logging.INFO)
      LOG.setLevel(log_levels.__next__())
      LOG.warning('SIGHUP received, switching to %s', logging._levelToName[LOG.level])
    elif _signum == signal.SIGINFO:
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
  for sig in (signal.SIGHUP, signal.SIGINFO, signal.SIGINT,
              signal.SIGQUIT, signal.SIGTERM):
    signal.signal(sig, _sig_handler)

  # Monitor the running threads.
  # Restart reconnect to a different cluster if a cluster thread dies.
  while s_thread.running():
    thread_list = [t for t in thread_enum() if isinstance(t, Cluster)]
    if len(thread_list) >= config.nb_threads:
      time.sleep(1)
      continue
    name, port = next_server()
    th_cluster = Cluster(name, port, queue, config.call, config.email)
    th_cluster.name = name
    th_cluster.daemon = True
    th_cluster.start()

  # stopping all the telnet threads
  for cth in (c for c in thread_enum() if isinstance(c, Cluster)):
    LOG.info('Stopping thread: %s', cth.name)
    cth.stop()
  s_thread.stop()


if __name__ == "__main__":
  main()
