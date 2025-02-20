#
# Copyright (c) 2023-2024, Fred C.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
#

import logging
import os
from pathlib import Path

import yaml

# The maximum amount of time we stay connected to one server.
TELNET_MAX_TIME = 3600 * 4
TELNET_TIMEOUT = 27

LOG = logging.getLogger('config')


class ConfigError(Exception):
  pass


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
    if self.config_data:
      return

    self.config_data = Config.read_config()
    if 'servers' not in self.config_data:
      raise ConfigError('Configuration error: "servers" key is missing')

    if isinstance(self.config_data['servers'], list):
      return

    server_file = Path(self.config_data['servers'])
    self.config_data['servers'] = self.read_servers(server_file)
    if len(self.config_data['servers']) < 1:
      raise ConfigError('Configuration error: server list empty')

  @staticmethod
  def read_servers(server_file):
    if not server_file.exists():
      raise FileNotFoundError(f'File "{server_file}" not found')
    server_list = []
    with server_file.open('r', encoding="utf-8") as fds:
      for line in (ln.strip() for ln in fds if not ln.startswith('#')):
        fields = [f.strip() for f in line.split(',')]
        if len(fields) != 4:
          LOG.warning('Incorrect number of fields: %s', line)
          continue
        try:
          server_list.append([fields[0], fields[1], int(fields[2])])
        except ValueError:
          LOG.warning('Server entry error: %s', line)

    return server_list

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
  def db_timeout(self):
    return self.config_data.get('db_timeout', 7)

  @property
  def nb_threads(self):
    return self.config_data.get('nb_threads', 2)

  @property
  def telnet_timeout(self):
    return self.config_data.get('telnet_timeout', TELNET_TIMEOUT)

  @property
  def telnet_max_time(self):
    return self.config_data.get('telnet_max_time', TELNET_MAX_TIME)

  @property
  def servers(self):
    return self.config_data['servers']

  @property
  def queue_size(self):
    return self.config_data.get('queue_size', 1024)

  @property
  def email(self):
    return self.config_data.get('email', f'{self.call}@arrl.org')

  @property
  def purge_time(self):
    return self.config_data.get('purge_time', 360)
