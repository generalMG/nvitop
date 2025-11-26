# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, ClassVar

from nvitop.tui.library import (
    NA,
    BufferedHistoryGraph,
    Device,
    GiB,
    HistoryGraph,
    MigDevice,
    NaType,
    bytes2human,
    colored,
    host,
    make_bar_chart,
    timedelta2human,
)
from nvitop.tui.screens.main.panels.base import BasePanel


if TYPE_CHECKING:
    import curses

    from nvitop.tui.tui import TUI


__all__ = ['HostPanel']


class HostPanel(BasePanel):  # pylint: disable=too-many-instance-attributes
    NAME: ClassVar[str] = 'host'
    SNAPSHOT_INTERVAL: ClassVar[float] = 0.5

    def __init__(
        self,
        devices: list[Device | MigDevice],
        compact: bool,
        *,
        win: curses.window | None,
        root: TUI,
    ) -> None:
        super().__init__(win, root)

        self.devices: list[Device | MigDevice] = devices
        self.device_count: int = len(self.devices)

        if win is not None:
            self.average_gpu_memory_percent: HistoryGraph | None = None
            self.average_gpu_utilization: HistoryGraph | None = None
            self.enable_history()

        self._compact: bool = compact
        self.width: int = max(79, root.width)
        self.full_height: int = 12
        self.compact_height: int = 3
        self.height: int = self.compact_height if compact else self.full_height

        self.cpu_percent: float = NA  # type: ignore[assignment]
        self.load_average: tuple[float, float, float] = (NA, NA, NA)  # type: ignore[assignment]
        self.virtual_memory: host.VirtualMemory = host.VirtualMemory()
        self.swap_memory: host.SwapMemory = host.SwapMemory()
        self.cpu_temperature: host.CpuTemperature = host.CpuTemperature(  # type: ignore[attr-defined]
            NA,
            NA,
            NA,
            None,
            None,
        )
        self.cpu_power: host.CpuPower = host.CpuPower(NA, NA, None)  # type: ignore[attr-defined]
        self._snapshot_daemon = threading.Thread(
            name='host-snapshot-daemon',
            target=self._snapshot_target,
            daemon=True,
        )
        self._daemon_running = threading.Event()

    @property
    def width(self) -> int:
        return self._width

    @width.setter
    def width(self, value: int) -> None:
        width = max(79, value)
        if self._width != width:
            if self.visible:
                self.need_redraw = True
            graph_width = max(width - 80, 20)
            if self.win is not None:
                self.average_gpu_memory_percent.width = graph_width  # type: ignore[union-attr]
                self.average_gpu_utilization.width = graph_width  # type: ignore[union-attr]
                for device in self.devices:
                    device.memory_percent.history.width = graph_width  # type: ignore[attr-defined]
                    device.gpu_utilization.history.width = graph_width  # type: ignore[attr-defined]
        self._width = width

    @property
    def compact(self) -> bool:
        return self._compact or self.no_unicode

    @compact.setter
    def compact(self, value: bool) -> None:
        value = value or self.no_unicode
        if self._compact != value:
            self.need_redraw = True
            self._compact = value
            self.height = self.compact_height if self.compact else self.full_height

    def enable_history(self) -> None:
        host.cpu_percent = BufferedHistoryGraph(
            interval=1.0,
            width=77,
            height=5,
            upsidedown=False,
            baseline=0.0,
            upperbound=100.0,
            dynamic_bound=False,
            format='CPU: {:.1f}%'.format,
        )(host.cpu_percent)
        host.virtual_memory = BufferedHistoryGraph(
            interval=1.0,
            width=77,
            height=4,
            upsidedown=True,
            baseline=0.0,
            upperbound=100.0,
            dynamic_bound=False,
            format='{:.1f}%'.format,
        )(host.virtual_memory, get_value=lambda vm: vm.percent)
        host.swap_memory = BufferedHistoryGraph(
            interval=1.0,
            width=77,
            height=1,
            upsidedown=False,
            baseline=0.0,
            upperbound=100.0,
            dynamic_bound=False,
            format='{:.1f}%'.format,
        )(host.swap_memory, get_value=lambda sm: sm.percent)

        def percentage(x: float | NaType) -> str:
            return f'{x:.1f}%' if x is not NA else NA

        def enable_history(device: Device) -> None:
            device.memory_percent = BufferedHistoryGraph(  # type: ignore[method-assign]
                interval=1.0,
                width=20,
                height=5,
                upsidedown=False,
                baseline=0.0,
                upperbound=100.0,
                dynamic_bound=False,
                format=lambda x: f'GPU {device.display_index} MEM: {percentage(x)}',
            )(device.memory_percent)
            device.gpu_utilization = BufferedHistoryGraph(  # type: ignore[method-assign]
                interval=1.0,
                width=20,
                height=5,
                upsidedown=True,
                baseline=0.0,
                upperbound=100.0,
                dynamic_bound=False,
                format=lambda x: f'GPU {device.display_index} UTL: {percentage(x)}',
            )(device.gpu_utilization)

        for device in self.devices:
            enable_history(device)

        prefix = 'AVG ' if self.device_count > 1 else ''
        self.average_gpu_memory_percent = BufferedHistoryGraph(
            interval=1.0,
            width=20,
            height=5,
            upsidedown=False,
            baseline=0.0,
            upperbound=100.0,
            dynamic_bound=False,
            format=lambda x: f'{prefix}GPU MEM: {percentage(x)}',
        )
        self.average_gpu_utilization = BufferedHistoryGraph(
            interval=1.0,
            width=20,
            height=5,
            upsidedown=True,
            baseline=0.0,
            upperbound=100.0,
            dynamic_bound=False,
            format=lambda x: f'{prefix}GPU UTL: {percentage(x)}',
        )

    @classmethod
    def set_snapshot_interval(cls, interval: float) -> None:
        assert interval > 0.0
        interval = float(interval)

        cls.SNAPSHOT_INTERVAL = min(interval / 3.0, 0.5)

    def take_snapshots(self) -> None:
        host.cpu_percent()
        host.virtual_memory()
        host.swap_memory()
        self.load_average = host.load_average()
        self.cpu_temperature = host.cpu_temperature()  # type: ignore[attr-defined]
        self.cpu_power = host.cpu_power_usage()  # type: ignore[attr-defined]

        self.cpu_percent = host.cpu_percent.history.last_value
        self.virtual_memory = host.virtual_memory.history.last_retval  # type: ignore[attr-defined]
        self.swap_memory = host.swap_memory.history.last_retval  # type: ignore[attr-defined]

        total_memory_used = 0
        total_memory_total = 0
        gpu_utilizations = []
        for device in self.devices:
            memory_used = device.snapshot.memory_used
            memory_total = device.snapshot.memory_total
            gpu_utilization = device.snapshot.gpu_utilization
            if memory_used is not NA and memory_total is not NA:
                total_memory_used += memory_used
                total_memory_total += memory_total
            if gpu_utilization is not NA:
                gpu_utilizations.append(float(gpu_utilization))
        if total_memory_total > 0:
            avg = 100.0 * total_memory_used / total_memory_total
            self.average_gpu_memory_percent.add(avg)  # type: ignore[union-attr]
        if len(gpu_utilizations) > 0:
            avg = sum(gpu_utilizations) / len(gpu_utilizations)
            self.average_gpu_utilization.add(avg)  # type: ignore[union-attr]

    def _format_temperature_bar(self) -> tuple[float | NaType, str]:
        """Return (percent, text) tuple for CPU temperature bar."""
        temperature = self.cpu_temperature
        percent: float | NaType = temperature.percent  # type: ignore[attr-defined]
        if percent is NA and temperature.current not in (NA, None):
            try:
                percent = float(temperature.current)
            except (TypeError, ValueError):
                percent = NA

        text = 'N/A'
        if temperature.current not in (NA, None):
            text = f'{float(temperature.current):.1f}C'
            limit = getattr(temperature, 'limit', NA)
            if limit not in (NA, None):
                try:
                    text += f'/{float(limit):.0f}C'
                except (TypeError, ValueError):
                    pass
        return percent, text

    def _format_power_bar(self) -> tuple[float | NaType, str]:
        """Return (percent, text) tuple for CPU power bar."""
        power = self.cpu_power
        percent: float | NaType = power.percent  # type: ignore[attr-defined]
        if percent is NA and power.watts not in (NA, None):
            try:
                percent = float(power.watts)
            except (TypeError, ValueError):
                percent = NA

        text = 'N/A'
        if power.watts not in (NA, None):
            try:
                text = f'{float(power.watts):.1f}W'
            except (TypeError, ValueError):
                text = 'N/A'
            limit = getattr(power, 'limit', NA)
            if limit not in (NA, None):
                try:
                    text += f'/{float(limit):.0f}W'
                except (TypeError, ValueError):
                    pass
        return percent, text

    @staticmethod
    def _attach_right_label(bar: str, value: str, width: int) -> str:
        """Attach a right-aligned value to a bar, trimming if needed."""
        value = value.strip()
        if not value:
            return bar
        if len(value) >= width:
            return value[:width].ljust(width)
        head = bar[: max(0, width - len(value) - 1)]
        return (head.rstrip() + ' ' + value).ljust(width)

    def _snapshot_target(self) -> None:
        self._daemon_running.wait()
        while self._daemon_running.is_set():
            self.take_snapshots()
            time.sleep(self.SNAPSHOT_INTERVAL)

    def frame_lines(self, compact: bool | None = None) -> list[str]:
        if compact is None:
            compact = self.compact
        if compact or self.no_unicode:
            return []

        remaining_width = self.width - 79
        data_line = (
            '│                                                                             │'
        )
        separator_line = (
            '├────────────╴120s├─────────────────────────╴60s├──────────╴30s├──────────────┤'
        )
        if self.width >= 100:
            data_line += ' ' * (remaining_width - 1) + '│'
            separator_line = separator_line[:-1] + '┼' + '─' * (remaining_width - 1) + '┤'

        frame = [
            '╞═══════════════════════════════╧══════════════════════╧══════════════════════╡',
            data_line,
            data_line,
            data_line,
            data_line,
            data_line,
            separator_line,
            data_line,
            data_line,
            data_line,
            data_line,
            data_line,
            '╘═════════════════════════════════════════════════════════════════════════════╛',
        ]
        if self.width >= 100:
            frame[0] = frame[0][:-1] + '╪' + '═' * (remaining_width - 1) + '╡'
            frame[-1] = frame[-1][:-1] + '╧' + '═' * (remaining_width - 1) + '╛'

        return frame

    def poke(self) -> None:
        if not self._daemon_running.is_set():
            self._daemon_running.set()
            self._snapshot_daemon.start()
            self.take_snapshots()

        super().poke()

    def draw(self) -> None:  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        self.color_reset()

        load_average = 'Load Average: {} {} {}'.format(
            *(f'{value:5.2f}'[:5] if value < 10000.0 else '9999+' for value in self.load_average),
        )
        temp_percent, temp_text = self._format_temperature_bar()
        power_percent, power_text = self._format_power_bar()
        tmp_color = 'yellow'
        try:
            tmp_value = float(temp_percent)
            if tmp_value >= 85.0:
                tmp_color = 'red'
            elif tmp_value >= 70.0:
                tmp_color = 'yellow'
            else:
                tmp_color = 'green'
        except Exception:  # noqa: BLE001
            pass

        if self.compact:
            width_right = len(load_average) + 4
            width_left = self.width - 2 - width_right
            cpu_bar = '[ {} ]'.format(
                make_bar_chart(
                    'CPU',
                    self.cpu_percent,
                    width_left - 4,
                    extra_text=f'  UPTIME: {timedelta2human(host.uptime(), round=True)}',
                ),
            )
            memory_bar = '[ {} ]'.format(
                make_bar_chart(
                    'MEM',
                    self.virtual_memory.percent,
                    width_left - 4,
                    extra_text=f'  USED: {bytes2human(self.virtual_memory.used, min_unit=GiB)}',
                ),
            )
            swap_bar = '[ {} ]'.format(
                make_bar_chart(
                    'SWP',
                    self.swap_memory.percent,
                    width_right - 4,
                ),
            )
            temp_bar = '[ {} ]'.format(
                self._attach_right_label(
                    make_bar_chart('TMP', temp_percent, width_left - 4, extra_text=''),
                    temp_text,
                    width_left - 4,
                ),
            )
            power_bar = '[ {} ]'.format(
                self._attach_right_label(
                    make_bar_chart('PWR', power_percent, width_right - 4, extra_text=''),
                    power_text,
                    width_right - 4,
                ),
            )
            self.addstr(self.y, self.x, f'{cpu_bar}  ( {load_average} )')
            self.addstr(self.y + 1, self.x, f'{memory_bar}  {swap_bar}')
            self.addstr(self.y + 2, self.x, f'{temp_bar}  {power_bar}')
            self.color_at(self.y, self.x, width=len(cpu_bar), fg='cyan', attr='bold')
            self.color_at(self.y + 1, self.x, width=width_left, fg='magenta', attr='bold')
            self.color_at(self.y, self.x + width_left + 2, width=width_right, attr='bold')
            self.color_at(
                self.y + 1,
                self.x + width_left + 2,
                width=width_right,
                fg='blue',
                attr='bold',
            )
            self.color_at(self.y + 2, self.x, width=width_left, fg=tmp_color, attr='bold')
            self.color_at(
                self.y + 2,
                self.x + width_left + 2,
                width=width_right,
                fg='red',
                attr='bold',
            )
            return

        remaining_width = self.width - 79

        if self.need_redraw:
            for y, line in enumerate(self.frame_lines(), start=self.y - 1):
                self.addstr(y, self.x, line)
            self.color_at(self.y + 5, self.x + 14, width=4, attr='dim')
            self.color_at(self.y + 5, self.x + 45, width=3, attr='dim')
            self.color_at(self.y + 5, self.x + 60, width=3, attr='dim')

            if self.width >= 100:
                for offset, string in (
                    (20, '╴30s├'),
                    (35, '╴60s├'),
                    (66, '╴120s├'),
                    (96, '╴180s├'),
                    (126, '╴240s├'),
                    (156, '╴300s├'),
                ):
                    if offset > remaining_width:
                        break
                    self.addstr(self.y + 5, self.x + self.width - offset, string)
                    self.color_at(
                        self.y + 5,
                        self.x + self.width - offset + 1,
                        width=len(string) - 2,
                        attr='dim',
                    )

        self.color(fg='cyan')
        for y, line in enumerate(host.cpu_percent.history.graph, start=self.y):
            self.addstr(y, self.x + 1, line)

        self.color(fg='magenta')
        for y, line in enumerate(host.virtual_memory.history.graph, start=self.y + 6):  # type: ignore[attr-defined]
            self.addstr(y, self.x + 1, line)

        self.color(fg='blue')
        for y, line in enumerate(host.swap_memory.history.graph, start=self.y + 10):  # type: ignore[attr-defined]
            self.addstr(y, self.x + 1, line)

        if self.width >= 100:
            if self.device_count > 1 and self.parent.selection.is_set():
                device = self.parent.selection.process.device  # type: ignore[union-attr]
                gpu_memory_percent = device.memory_percent.history  # type: ignore[union-attr]
                gpu_utilization = device.gpu_utilization.history  # type: ignore[union-attr]
            else:
                gpu_memory_percent = self.average_gpu_memory_percent
                gpu_utilization = self.average_gpu_utilization

            if self.TERM_256COLOR:
                for i, (y, line) in enumerate(enumerate(gpu_memory_percent.graph, start=self.y)):
                    self.addstr(y, self.x + 79, line, self.get_fg_bg_attr(fg=1.0 - i / 4.0))

                for i, (y, line) in enumerate(enumerate(gpu_utilization.graph, start=self.y + 6)):
                    self.addstr(y, self.x + 79, line, self.get_fg_bg_attr(fg=i / 4.0))
            else:
                self.color(fg=Device.color_of(gpu_memory_percent.last_value, type='memory'))
                for y, line in enumerate(gpu_memory_percent.graph, start=self.y):
                    self.addstr(y, self.x + 79, line)

                self.color(fg=Device.color_of(gpu_utilization.last_value, type='gpu'))
                for y, line in enumerate(gpu_utilization.graph, start=self.y + 6):
                    self.addstr(y, self.x + 79, line)

        self.color_reset()
        self.addstr(self.y, self.x + 1, f' {load_average} ')
        self.addstr(self.y + 1, self.x + 1, f' {host.cpu_percent.history} ')
        inner_width = self.width - 2
        host_inner_width = getattr(host.cpu_percent.history, 'width', 77)
        mini_total_width = min(host_inner_width, max(24, min(50, host_inner_width)))
        temp_width = mini_total_width // 2
        power_width = max(10, mini_total_width - temp_width - 2)
        tmp_line = ' ' + self._attach_right_label(
            make_bar_chart('TMP', temp_percent, temp_width, extra_text=''),
            temp_text,
            temp_width,
        )
        pwr_line = ' ' + self._attach_right_label(
            make_bar_chart('PWR', power_percent, power_width, extra_text=''),
            power_text,
            power_width,
        )
        if self.width >= 100:
            self.addstr(self.y + 2, self.x + 1, tmp_line.ljust(host_inner_width))
            self.addstr(self.y + 3, self.x + 1, pwr_line.ljust(host_inner_width))
            self.addstr(self.y + 2, self.x + 78, '│')
            self.addstr(self.y + 3, self.x + 78, '│')
        else:
            self.addstr(self.y + 2, self.x + 1, tmp_line.ljust(inner_width))
            self.addstr(self.y + 3, self.x + 1, pwr_line.ljust(inner_width))
        self.color_at(self.y + 2, self.x + 1, width=temp_width, fg=tmp_color, attr='bold')
        self.color_at(self.y + 3, self.x + 1, width=temp_width, fg=tmp_color, attr='bold')
        self.color_at(
            self.y + 2,
            self.x + 1 + temp_width + 2,
            width=power_width,
            fg='red',
            attr='bold',
        )
        self.color_at(
            self.y + 3,
            self.x + 1 + temp_width + 2,
            width=power_width,
            fg='red',
            attr='bold',
        )
        self.addstr(
            self.y + 9,
            self.x + 1,
            (
                f' MEM: {bytes2human(self.virtual_memory.used, min_unit=GiB)} '
                f'({host.virtual_memory.history}) '  # type: ignore[attr-defined]
            ),
        )
        self.addstr(
            self.y + 10,
            self.x + 1,
            (
                f' SWP: {bytes2human(self.swap_memory.used, min_unit=GiB)} '
                f'({host.swap_memory.history}) '  # type: ignore[attr-defined]
            ),
        )
        if self.width >= 100:
            self.addstr(self.y, self.x + 79, f' {gpu_memory_percent} ')
            self.addstr(self.y + 10, self.x + 79, f' {gpu_utilization} ')

    def destroy(self) -> None:
        super().destroy()
        self._daemon_running.clear()

    def print_width(self) -> int:
        if self.device_count > 0 and self.width >= 100:
            return self.width
        return 79

    def print(self) -> None:
        self.cpu_percent = host.cpu_percent()
        self.virtual_memory = host.virtual_memory()
        self.swap_memory = host.swap_memory()
        self.load_average = host.load_average()
        self.cpu_temperature = host.cpu_temperature()  # type: ignore[attr-defined]
        self.cpu_power = host.cpu_power_usage()  # type: ignore[attr-defined]

        load_average = 'Load Average: {} {} {}'.format(
            *(f'{value:5.2f}'[:5] if value < 10000.0 else '9999+' for value in self.load_average),
        )
        temp_percent, temp_text = self._format_temperature_bar()
        power_percent, power_text = self._format_power_bar()
        tmp_color = 'yellow'
        try:
            tmp_value = float(temp_percent)
            if tmp_value >= 85.0:
                tmp_color = 'red'
            elif tmp_value >= 70.0:
                tmp_color = 'yellow'
            else:
                tmp_color = 'green'
        except Exception:  # noqa: BLE001
            pass

        width_right = len(load_average) + 4
        width_left = self.width - 2 - width_right
        cpu_bar = '[ {} ]'.format(
            make_bar_chart(
                'CPU',
                self.cpu_percent,
                width_left - 4,
                extra_text=f'  UPTIME: {timedelta2human(host.uptime(), round=True)}',
            ),
        )
        memory_bar = '[ {} ]'.format(
            make_bar_chart(
                'MEM',
                self.virtual_memory.percent,
                width_left - 4,
                extra_text=f'  USED: {bytes2human(self.virtual_memory.used, min_unit=GiB)}',
            ),
        )
        swap_bar = '[ {} ]'.format(make_bar_chart('SWP', self.swap_memory.percent, width_right - 4))
        temp_bar = '[ {} ]'.format(
            self._attach_right_label(
                make_bar_chart('TMP', temp_percent, width_left - 4, extra_text=''),
                temp_text,
                width_left - 4,
            ),
        )
        power_bar = '[ {} ]'.format(
            self._attach_right_label(
                make_bar_chart('PWR', power_percent, width_right - 4, extra_text=''),
                power_text,
                width_right - 4,
            ),
        )

        lines = [
            '{}  {}'.format(
                colored(cpu_bar, color='cyan', attrs=('bold',)),
                colored(f'( {load_average} )', attrs=('bold',)),
            ),
            '{}  {}'.format(
                colored(memory_bar, color='magenta', attrs=('bold',)),
                colored(swap_bar, color='blue', attrs=('bold',)),
            ),
            '{}  {}'.format(
                colored(temp_bar, color=tmp_color, attrs=('bold',)),
                colored(power_bar, color='red', attrs=('bold',)),
            ),
        ]

        lines = '\n'.join(lines)
        if self.no_unicode:
            lines = lines.translate(self.ASCII_TRANSTABLE)

        try:
            print(lines)
        except UnicodeError:
            print(lines.translate(self.ASCII_TRANSTABLE))

    def press(self, key: int) -> bool:
        self.root.keymaps.use_keymap('host')
        return self.root.press(key)
