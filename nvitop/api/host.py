# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
#
# Copyright 2021-2025 Xuehai Pan. All Rights Reserved.
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
# ==============================================================================
"""Shortcuts for package ``psutil``.

``psutil`` is a cross-platform library for retrieving information on running processes and system
utilization (CPU, memory, disks, network, sensors) in Python.
"""

from __future__ import annotations

import os as _os
import pathlib as _pathlib
import threading as _threading
import time as _time
from typing import TYPE_CHECKING, NamedTuple

import psutil as _psutil
from psutil import *  # noqa: F403 # pylint: disable=wildcard-import,unused-wildcard-import,redefined-builtin
from psutil import (  # noqa: F401
    LINUX,
    MACOS,
    POSIX,
    WINDOWS,
    AccessDenied,
    Error,
    NoSuchProcess,
    Process,
    ZombieProcess,
    boot_time,
    cpu_percent,
    pids,
    swap_memory,
    virtual_memory,
)
from psutil import Error as PsutilError  # pylint: disable=reimported

from nvitop.api.utils import NA, NaType


__all__ = [
    'WINDOWS_SUBSYSTEM_FOR_LINUX',
    'WSL',
    'PsutilError',
    'CpuPower',
    'CpuTemperature',
    'cpu_power_usage',
    'cpu_temperature',
    'getuser',
    'hostname',
    'load_average',
    'memory_percent',
    'ppid_map',
    'reverse_ppid_map',
    'swap_percent',
    'uptime',
]
__all__ += [name for name in _psutil.__all__ if not name.startswith('_') and name != 'Error']


del Error  # renamed to PsutilError


def getuser() -> str:
    """Get the current username from the environment or password database."""
    import getpass  # pylint: disable=import-outside-toplevel

    try:
        return getpass.getuser()
    except (ModuleNotFoundError, OSError):
        return _os.getlogin()


def hostname() -> str:
    """Get the hostname of the machine."""
    import platform  # pylint: disable=import-outside-toplevel

    return platform.node()


if hasattr(_psutil, 'getloadavg'):

    def load_average() -> tuple[float, float, float]:
        """Get the system load average."""
        return _psutil.getloadavg()

else:

    def load_average() -> None:  # type: ignore[misc]
        """Get the system load average."""
        return


def uptime() -> float:
    """Get the system uptime."""
    import time as _time  # pylint: disable=import-outside-toplevel

    return _time.time() - boot_time()


def memory_percent() -> float:
    """The percentage usage of virtual memory, calculated as ``(total - available) / total * 100``."""
    return virtual_memory().percent


def swap_percent() -> float:
    """The percentage usage of virtual memory, calculated as ``used / total * 100``."""
    return swap_memory().percent


def ppid_map() -> dict[int, int]:
    """Obtain a ``{pid: ppid, ...}`` dict for all running processes in one shot."""
    ret = {}
    for pid in pids():
        try:
            ret[pid] = Process(pid).ppid()
        except (NoSuchProcess, ZombieProcess):  # noqa: PERF203
            pass
    return ret


try:
    from psutil import _ppid_map as ppid_map  # type: ignore[no-redef] # noqa: F811,RUF100
except ImportError:
    pass


def reverse_ppid_map() -> dict[int, list[int]]:
    """Obtain a ``{ppid: [pid, ...], ...}`` dict for all running processes in one shot."""
    from collections import defaultdict  # pylint: disable=import-outside-toplevel

    ret = defaultdict(list)
    for pid, ppid in ppid_map().items():
        ret[ppid].append(pid)

    return ret


# pylint: disable=invalid-name
if LINUX:
    WSL = _os.getenv('WSL_DISTRO_NAME', default=None)
    if WSL is not None and WSL == '':
        WSL = 'WSL'
else:
    WSL = None
WINDOWS_SUBSYSTEM_FOR_LINUX = WSL
"""The Linux distribution name of the Windows Subsystem for Linux."""
# pylint: enable=invalid-name


class CpuTemperature(NamedTuple):  # pylint: disable=missing-class-docstring
    current: float | NaType
    high: float | NaType
    critical: float | NaType
    label: str | None = None
    sensor: str | None = None

    @property
    def limit(self) -> float | NaType:
        """Return the temperature limit used for normalization."""
        if self.high not in (None, NA):
            return self.high
        if self.critical not in (None, NA):
            return self.critical
        return NA

    @property
    def percent(self) -> float | NaType:
        """Return the temperature as a percentage of the limit."""
        try:
            limit = float(self.limit)
            current = float(self.current)
        except (TypeError, ValueError):
            return NA
        if not (limit > 0.0):
            return NA
        return 100.0 * current / limit


class CpuPower(NamedTuple):  # pylint: disable=missing-class-docstring
    watts: float | NaType
    limit: float | NaType
    source: str | None = None

    @property
    def percent(self) -> float | NaType:
        """Return the power draw as a percentage of the power limit (if available)."""
        try:
            limit = float(self.limit)
            watts = float(self.watts)
        except (TypeError, ValueError):
            return NA
        if not (limit > 0.0):
            return NA
        return 100.0 * watts / limit


CPU_TEMP_HINTS: tuple[str, ...] = (
    'cpu',
    'coretemp',
    'k10temp',
    'zenpower',
    'package',
    'soc',
    'tctl',
    'tdie',
    'tcpu',
    'acpitz',
)


def _looks_like_cpu_sensor(name: str, label: str | None) -> bool:
    lname = name.lower()
    if any(token in lname for token in CPU_TEMP_HINTS):
        return True
    if label:
        llabel = label.lower()
        if any(token in llabel for token in CPU_TEMP_HINTS):
            return True
    return False


def _safe_int(path: _pathlib.Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (FileNotFoundError, PermissionError, ValueError, OSError):
        return None


def _safe_str(path: _pathlib.Path) -> str | None:
    try:
        return path.read_text().strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None


def cpu_temperature() -> CpuTemperature:
    """Return the current CPU temperature (best effort).

    The hottest CPU reading is selected from available sensors. If no CPU sensor is found or
    sensors are unavailable, :const:`NA` fields are returned.
    """
    if not hasattr(_psutil, 'sensors_temperatures'):
        return CpuTemperature(NA, NA, NA, None, None)

    try:
        temperatures = _psutil.sensors_temperatures(fahrenheit=False)  # type: ignore[attr-defined]
    except (AttributeError, NotImplementedError, PsutilError):
        return CpuTemperature(NA, NA, NA, None, None)

    cpu_readings: list[CpuTemperature] = []
    fallback_readings: list[CpuTemperature] = []
    for name, entries in temperatures.items():
        if not entries:
            continue
        for entry in entries:
            current = entry.current
            if current is None:
                continue
            reading = CpuTemperature(
                current=float(current),
                high=float(entry.high) if entry.high is not None else NA,
                critical=float(entry.critical) if entry.critical is not None else NA,
                label=entry.label or None,
                sensor=name,
            )
            if _looks_like_cpu_sensor(name, entry.label):
                cpu_readings.append(reading)
            else:
                fallback_readings.append(reading)

    readings = cpu_readings or fallback_readings
    if len(readings) == 0:
        return CpuTemperature(NA, NA, NA, None, None)
    return max(readings, key=lambda temp: float(temp.current))


class _RaplDomain:
    def __init__(self, path: _pathlib.Path) -> None:
        self.path = path
        self.energy_path = path / 'energy_uj'
        self.power_path = path / 'power_uw'
        self.max_energy = _safe_int(path / 'max_energy_range_uj')
        self.limit = self._read_power_limit()
        self.name = _safe_str(path / 'name')
        self._last_energy: int | None = None
        self._last_time: float | None = None

        if not self.power_path.is_file():
            self.power_path = None  # type: ignore[assignment]

    def _read_power_limit(self) -> float | None:
        for filename in (
            'constraint_0_power_limit_uw',
            'constraint_1_power_limit_uw',
            'max_power_range_uw',
            'power_limit_uw',
        ):
            value = _safe_int(self.path / filename)
            if value is not None:
                return value / 1e6  # microwatts -> watts
        return None

    def power(self, now: float) -> float | None:
        """Return instantaneous power in watts if available."""
        if self.power_path is not None:
            value = _safe_int(self.power_path)
            if value is not None:
                return value / 1e6

        energy = _safe_int(self.energy_path)
        if energy is None:
            return None
        if self._last_energy is None or self._last_time is None:
            self._last_energy = energy
            self._last_time = now
            return None

        delta_energy = energy - self._last_energy
        if delta_energy < 0 and self.max_energy:
            delta_energy += self.max_energy
        self._last_energy = energy
        elapsed = now - self._last_time
        self._last_time = now
        if elapsed <= 0:
            return None
        return delta_energy / elapsed / 1e6


class _CpuPowerSensor:
    def __init__(self) -> None:
        self._domains: list[_RaplDomain] = self._discover_domains()
        self._lock = _threading.Lock()
        self._observed_peak: float = 0.0

    def _discover_domains(self) -> list[_RaplDomain]:
        base = _pathlib.Path('/sys/class/powercap')
        if not base.is_dir():
            return []

        domains: list[_RaplDomain] = []
        for path in sorted(base.iterdir()):
            name = path.name
            if not name.startswith(('intel-rapl:', 'amd-rapl:')):
                continue
            rapl_name = _safe_str(path / 'name')
            if rapl_name is not None and 'psys' in rapl_name.lower():
                # System-level power; skip to focus on package domains.
                continue
            if (path / 'energy_uj').is_file() or (path / 'power_uw').is_file():
                domains.append(_RaplDomain(path))
        return domains

    def read(self) -> CpuPower | None:
        now = _time.monotonic()
        with self._lock:
            total_power = 0.0
            has_power = False
            total_limit = 0.0
            has_limit = False

            for domain in self._domains:
                power = domain.power(now)
                if power is not None:
                    total_power += power
                    has_power = True
                if domain.limit is not None:
                    total_limit += domain.limit
                    has_limit = True

            if not has_power:
                return None

            if has_limit:
                limit = total_limit
            else:
                self._observed_peak = max(self._observed_peak, total_power)
                limit = self._observed_peak * 1.25 if self._observed_peak > 0 else None

            return CpuPower(
                watts=total_power,
                limit=limit if limit is not None else NA,
                source='rapl' if self._domains else None,
            )


_CPU_POWER_SENSOR = _CpuPowerSensor()


def cpu_power_usage() -> CpuPower:
    """Return the current CPU package power draw using RAPL if available."""
    reading = _CPU_POWER_SENSOR.read()
    if reading is None:
        return CpuPower(NA, NA, None)
    return reading
