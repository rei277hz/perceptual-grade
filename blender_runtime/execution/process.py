"""Explicit installed-package bootstrap with an injectable test launcher."""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import time

from .contracts import PROTOCOL
from .transport import Channel


def fingerprint(package):
    digest = hashlib.sha256()
    for directory in ('.',):
        for path in sorted((Path(package) / directory).rglob('*')):
            if path.is_file() and path.suffix in ('.py', '.npz', '.json'):
                digest.update(path.relative_to(package).as_posix().encode())
                digest.update(path.read_bytes())
    return digest.hexdigest()


def launch_command(config):
    """Default headless Python host; applications inject another launcher."""
    return [sys.executable, config['worker_entry'], config['config_path']]


class Helper:
    def __init__(self, session, role, generation, package, build, launcher=None, *, profile=None, worker_host=None, host_settings=None, environment=None):
        self.role, self.generation = role, generation
        self.channel = None
        self.ready = False
        self.active = None
        self.started = self.last_progress = time.monotonic()
        self.token = secrets.token_hex(32)
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.bind(('127.0.0.1', 0))
        self.listener.listen(1)
        self.listener.setblocking(False)
        config_path = session.path / f'helper-{role}-{generation}.json'
        config = dict(protocol=PROTOCOL, session=session.identity, directory=str(session.path),
                      role=role, generation=generation, token=self.token, package=str(package),
                      fingerprint=build, address=self.listener.getsockname(),
                      worker_entry=str(Path(__file__).with_name('worker_main.py').absolute()),
                      worker_host=worker_host, host_settings=host_settings or {},
                      config_path=str(config_path))
        if profile is not None:
            config['profile'] = asdict(profile)
        config_path.write_text(json.dumps(config), encoding='utf8')
        self.log_path = session.path / f'helper-{role}-{generation}.log'
        self.log = self.log_path.open('wb')
        env = os.environ.copy() if environment is None else dict(environment)
        env['OPENBLAS_NUM_THREADS'] = '1'
        env['OMP_NUM_THREADS'] = '1'
        try:
            self.process = subprocess.Popen((launcher or launch_command)(config), env=env,
                                            stdin=subprocess.DEVNULL, stdout=self.log,
                                            stderr=subprocess.STDOUT, close_fds=True)
        except BaseException:
            self.listener.close()
            self.log.close()
            raise

    def poll(self):
        if self.channel is None:
            try:
                sock, _ = self.listener.accept()
            except BlockingIOError:
                return []
            self.channel = Channel(sock)
            self.listener.close()
        return self.channel.pump()

    def send(self, message):
        self.channel.send(message)

    def terminate(self):
        if self.process.poll() is None:
            self.process.terminate()

    def kill(self):
        if self.process.poll() is None:
            self.process.kill()

    def close(self):
        if self.channel is not None:
            self.channel.close()
        self.listener.close()
        self.log.close()

    def error(self):
        try:
            return self.log_path.read_text(errors='replace')[-3000:]
        except OSError:
            return ''
