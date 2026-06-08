"""
moses/industrial/plc.py
=======================

PLC integration for humanoid robot cell control.

Supports:
  - Modbus TCP/RTU (IEC 61784-1, CPF 15)
  - OPC-UA (IEC 62541)
  - EtherNet/IP (CIP, IEC 61158)

Safety I/O:
  - E-stop chains (ISO 13850)
  - Light curtains (IEC 61496)
  - Safety door locks (ISO 14119)

Program structure follows IEC 61131-3 concepts:
  - Program Organisation Units (POUs)
  - Function Blocks for reusable logic
  - Ladder logic rung abstraction

Author: Moses Industrial Team
Version: 6.0.0
"""

from __future__ import annotations

import asyncio
import struct
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union


# ---------------------------------------------------------------------------
# Common types & constants
# ---------------------------------------------------------------------------

class SafetyState(Enum):
    """Safety state machine per ISO 13849-1."""
    INIT = auto()
    SAFE = auto()
    WARNING = auto()
    FAULT = auto()
    ESTOP_ACTIVE = auto()


class PLCProtocol(Enum):
    MODBUS_TCP = "modbus_tcp"
    MODBUS_RTU = "modbus_rtu"
    OPC_UA = "opc_ua"
    ETHERNET_IP = "ethernet_ip"


@dataclass(frozen=True)
class IOAddress:
    """Unified I/O address descriptor."""
    protocol: PLCProtocol
    rack: int = 0
    slot: int = 0
    byte_offset: int = 0
    bit_offset: int = 0
    data_type: str = "BOOL"          # BOOL, BYTE, WORD, DWORD, REAL, etc.

    def __str__(self) -> str:
        return f"{self.protocol.value}://{self.rack}/{self.slot}.{self.byte_offset}.{self.bit_offset}"


# ---------------------------------------------------------------------------
# Digital / Analog I/O mapping
# ---------------------------------------------------------------------------

@dataclass
class DigitalIOMap:
    """
    Digital I/O mapping table.

    Maps symbolic names (e.g. 'gripper_open') to physical I/O addresses.
    Supports both inputs (%I) and outputs (%Q) in IEC 61131 notation.
    """
    name: str
    address: IOAddress
    direction: str = "input"         # "input" | "output"
    inverted: bool = False
    debounce_ms: float = 0.0
    _last_read: bool = field(default=False, repr=False)
    _last_time: float = field(default=0.0, repr=False)

    def read(self, raw_value: bool) -> bool:
        """Apply inversion and debounce."""
        now = time.monotonic() * 1000
        val = raw_value ^ self.inverted
        if self.debounce_ms > 0:
            if val != self._last_read:
                if now - self._last_time < self.debounce_ms:
                    return self._last_read
                self._last_time = now
        self._last_read = val
        return val


@dataclass
class AnalogIOMap:
    """Analog I/O mapping with scaling."""
    name: str
    address: IOAddress
    direction: str = "input"
    raw_min: int = 0
    raw_max: int = 32767
    eng_min: float = 0.0
    eng_max: float = 10.0
    unit: str = "V"

    def scale(self, raw: int) -> float:
        """Scale raw ADC value to engineering units."""
        ratio = (raw - self.raw_min) / (self.raw_max - self.raw_min)
        return self.eng_min + ratio * (self.eng_max - self.eng_min)

    def unscale(self, eng: float) -> int:
        """Convert engineering units back to raw value."""
        ratio = (eng - self.eng_min) / (self.eng_max - self.eng_min)
        return int(self.raw_min + ratio * (self.raw_max - self.raw_min))


# ---------------------------------------------------------------------------
# Safety I/O
# ---------------------------------------------------------------------------

@dataclass
class SafetyIOMap:
    """
    Safety I/O abstraction.

    Implements dual-channel safety inputs per ISO 13849-1 Category 3/4.
    Safety outputs are monitored (EDM - External Device Monitoring).
    """
    name: str
    channel_a: IOAddress
    channel_b: IOAddress
    output: Optional[IOAddress] = None
    edm_feedback: Optional[IOAddress] = None
    category: int = 3                # ISO 13849-1 Category (1..4)
    pl_target: str = "d"             # Target Performance Level a..e

    def evaluate(self, val_a: bool, val_b: bool) -> Tuple[bool, Optional[str]]:
        """
        Evaluate dual-channel safety input.

        Returns (safe_state_ok, fault_reason).
        For Category 3/4, both channels must agree (normally closed).
        """
        if self.category >= 3:
            if val_a != val_b:
                return False, f"Discrepancy fault on {self.name}: A={val_a} B={val_b}"
        # For Category 1/2, single channel is sufficient
        safe = val_a and val_b
        return safe, None


class EStopChain:
    """
    E-stop chain handler per ISO 13850.

    Monitors multiple e-stop buttons in series (normally closed).
    Any opened contact triggers safe state.
    """
    def __init__(self, name: str = "estop_chain") -> None:
        self.name = name
        self.stops: List[SafetyIOMap] = []
        self.state = SafetyState.SAFE

    def add_estop(self, safety_io: SafetyIOMap) -> None:
        self.stops.append(safety_io)

    def evaluate(self, io_values: Dict[str, Tuple[bool, bool]]) -> SafetyState:
        """Evaluate entire e-stop chain."""
        for stop in self.stops:
            vals = io_values.get(stop.name, (True, True))
            ok, fault = stop.evaluate(vals[0], vals[1])
            if not ok:
                self.state = SafetyState.ESTOP_ACTIVE
                return self.state
        self.state = SafetyState.SAFE
        return self.state


class LightCurtain:
    """
    Safety light curtain per IEC 61496-1 (Type 4) / IEC 61496-2.

    OSSD (Output Signal Switching Device) dual-channel outputs.
    """
    def __init__(
        self,
        name: str,
        ossd1: IOAddress,
        ossd2: IOAddress,
        resolution_mm: float = 14.0,      # finger detection
        response_time_ms: float = 15.0,
    ) -> None:
        self.name = name
        self.ossd = SafetyIOMap(
            name=name,
            channel_a=ossd1,
            channel_b=ossd2,
            category=4,
            pl_target="e",
        )
        self.resolution = resolution_mm
        self.response_time = response_time_ms

    def calculate_safety_distance(self, approach_speed: float) -> float:
        """
        Calculate minimum safety distance per ISO 13855.

        S = K * T + C
        K = 1600 mm/s (hand speed, ISO 13855)
        T = response time of protective device + machine stopping time
        C = 850 mm (intrusion distance, light curtain resolution <= 40 mm)
        """
        K = 1600.0          # mm/s
        C = 850.0           # mm for resolution <= 40 mm
        # T must include machine stopping time (provided externally)
        # Here we use only the light curtain response time as example
        T = self.response_time / 1000.0
        S = K * T + C
        return S


class SafetyDoorLock:
    """
    Safety door interlock per ISO 14119.

    Supports both mechanical and non-contact (RFID) interlocks.
    """
    def __init__(
        self,
        name: str,
        locked_sensor: SafetyIOMap,
        closed_sensor: SafetyIOMap,
        lock_output: IOAddress,
        lock_type: str = "power_to_lock",    # "power_to_lock" | "power_to_unlock"
    ) -> None:
        self.name = name
        self.locked_sensor = locked_sensor
        self.closed_sensor = closed_sensor
        self.lock_output = lock_output
        self.lock_type = lock_type

    def is_safe_to_start(self, io_values: Dict[str, Tuple[bool, bool]]) -> bool:
        """Check if door is closed and locked before allowing operation."""
        closed_ok, _ = self.closed_sensor.evaluate(*io_values.get(self.closed_sensor.name, (True, True)))
        locked_ok, _ = self.locked_sensor.evaluate(*io_values.get(self.locked_sensor.name, (True, True)))
        return closed_ok and locked_ok


# ---------------------------------------------------------------------------
# Ladder logic concepts (IEC 61131-3)
# ---------------------------------------------------------------------------

@dataclass
class Rung:
    """
    Abstract ladder logic rung.

    A rung is a horizontal line of logic:
      |--[ ]--[ ]--( )--|
    Represented as a list of conditions and one coil (output).
    """
    name: str
    conditions: List[Callable[[], bool]] = field(default_factory=list)
    coil: Optional[Callable[[bool], None]] = None
    _state: bool = field(default=False, repr=False)

    def evaluate(self) -> bool:
        """Evaluate rung: all conditions ANDed together."""
        result = all(cond() for cond in self.conditions)
        self._state = result
        if self.coil:
            self.coil(result)
        return result


class FunctionBlock(Protocol):
    """IEC 61131-3 Function Block protocol."""
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]: ...


@dataclass
class TON_Timer:
    """On-delay timer (IEC 61131-3)."""
    name: str
    pt_ms: float = 1000.0
    _start_time: float = field(default=0.0, repr=False)
    _running: bool = field(default=False, repr=False)
    q: bool = field(default=False, repr=False)
    et: float = field(default=0.0, repr=False)

    def execute(self, inp: bool) -> None:
        now = time.monotonic() * 1000
        if inp and not self._running:
            self._running = True
            self._start_time = now
        elif not inp:
            self._running = False
            self.q = False
            self.et = 0.0
            return

        if self._running:
            elapsed = now - self._start_time
            self.et = min(elapsed, self.pt_ms)
            self.q = elapsed >= self.pt_ms


# ---------------------------------------------------------------------------
# Protocol clients
# ---------------------------------------------------------------------------

class PLCInterface(Protocol):
    """Abstract PLC interface."""
    async def connect(self) -> None: ...
    async def disconnect(self) -> None: ...
    async def read_digital(self, address: IOAddress) -> bool: ...
    async def write_digital(self, address: IOAddress, value: bool) -> None: ...
    async def read_register(self, address: IOAddress) -> int: ...
    async def write_register(self, address: IOAddress, value: int) -> None: ...


class ModbusClient:
    """
    Modbus TCP/RTU client.

    Uses pymodbus (install: pip install pymodbus).
    Register mapping per Modicon convention:
      - Coils (0x)      : digital outputs
      - Discrete Inputs (1x): digital inputs
      - Holding Registers (4x): analog outputs / parameters
      - Input Registers (3x): analog inputs
    """
    def __init__(
        self,
        host: str = "192.168.1.10",
        port: int = 502,
        protocol: str = "tcp",
        slave_id: int = 1,
        timeout: float = 1.0,
    ) -> None:
        self.host = host
        self.port = port
        self.protocol = protocol
        self.slave_id = slave_id
        self.timeout = timeout
        self._client: Any = None

    async def connect(self) -> None:
        from pymodbus.client import ModbusTcpClient, ModbusSerialClient
        if self.protocol == "tcp":
            self._client = ModbusTcpClient(self.host, port=self.port, timeout=self.timeout)
        else:
            self._client = ModbusSerialClient(
                port=self.host, baudrate=9600, parity="N", stopbits=1, bytesize=8
            )
        self._client.connect()

    async def disconnect(self) -> None:
        if self._client:
            self._client.close()
            self._client = None

    async def read_digital(self, address: IOAddress) -> bool:
        """Read coil (0x) or discrete input (1x)."""
        addr = address.byte_offset * 8 + address.bit_offset
        if address.direction == "input":
            rr = self._client.read_discrete_inputs(addr, 1, slave=self.slave_id)
        else:
            rr = self._client.read_coils(addr, 1, slave=self.slave_id)
        if rr.isError():
            raise RuntimeError(f"Modbus read error: {rr}")
        return bool(rr.bits[0])

    async def write_digital(self, address: IOAddress, value: bool) -> None:
        addr = address.byte_offset * 8 + address.bit_offset
        self._client.write_coil(addr, value, slave=self.slave_id)

    async def read_register(self, address: IOAddress) -> int:
        """Read holding (4x) or input (3x) register."""
        addr = address.byte_offset
        if address.direction == "input":
            rr = self._client.read_input_registers(addr, 1, slave=self.slave_id)
        else:
            rr = self._client.read_holding_registers(addr, 1, slave=self.slave_id)
        if rr.isError():
            raise RuntimeError(f"Modbus read error: {rr}")
        return rr.registers[0]

    async def write_register(self, address: IOAddress, value: int) -> None:
        addr = address.byte_offset
        self._client.write_register(addr, value, slave=self.slave_id)


class OPCUAClient:
    """
    OPC-UA client per IEC 62541.

    Uses asyncua (install: pip install asyncua).
    Supports browsing, reading, writing, and subscribing to nodes.
    """
    def __init__(
        self,
        endpoint: str = "opc.tcp://192.168.1.10:4840",
        namespace: str = "http://moses.robot/industrial",
    ) -> None:
        self.endpoint = endpoint
        self.namespace = namespace
        self._client: Any = None
        self._nsidx: int = 2

    async def connect(self) -> None:
        from asyncua import Client
        self._client = Client(url=self.endpoint)
        await self._client.connect()
        self._nsidx = await self._client.get_namespace_index(self.namespace)

    async def disconnect(self) -> None:
        if self._client:
            await self._client.disconnect()
            self._client = None

    def _nodeid(self, address: IOAddress) -> str:
        """Map IOAddress to OPC-UA node ID string."""
        return f"ns={self._nsidx};s={address.name}"

    async def read_digital(self, address: IOAddress) -> bool:
        node = self._client.get_node(self._nodeid(address))
        val = await node.read_value()
        return bool(val)

    async def write_digital(self, address: IOAddress, value: bool) -> None:
        node = self._client.get_node(self._nodeid(address))
        await node.write_value(value)

    async def read_register(self, address: IOAddress) -> int:
        node = self._client.get_node(self._nodeid(address))
        val = await node.read_value()
        return int(val)

    async def write_register(self, address: IOAddress, value: int) -> None:
        node = self._client.get_node(self._nodeid(address))
        await node.write_value(value)

    async def subscribe_data_change(
        self,
        address: IOAddress,
        callback: Callable[[Any, Any], None],
    ) -> Any:
        """Subscribe to data changes (events)."""
        handler = _SubscriptionHandler(callback)
        sub = await self._client.create_subscription(100, handler)
        node = self._client.get_node(self._nodeid(address))
        await sub.subscribe_data_change(node)
        return sub


class _SubscriptionHandler:
    def __init__(self, callback: Callable[[Any, Any], None]) -> None:
        self._cb = callback

    async def datachange_notification(self, node: Any, val: Any, data: Any) -> None:
        self._cb(node, val)


class EtherNetIPClient:
    """
    EtherNet/IP (CIP) client.

    Uses pycomm3 (install: pip install pycomm3).
    Supports explicit messaging and implicit (I/O) connections.
    """
    def __init__(
        self,
        host: str = "192.168.1.10",
        slot: int = 0,
        timeout: float = 5.0,
    ) -> None:
        self.host = host
        self.slot = slot
        self.timeout = timeout
        self._driver: Any = None

    async def connect(self) -> None:
        from pycomm3 import CIPDriver
        self._driver = CIPDriver(f"{self.host}/{self.slot}")
        self._driver.open()

    async def disconnect(self) -> None:
        if self._driver:
            self._driver.close()
            self._driver = None

    async def read_tag(self, tag_name: str) -> Any:
        """Read a CIP tag by symbolic name."""
        result = self._driver.read_tag(tag_name)
        if result.error:
            raise RuntimeError(f"CIP read error: {result.error}")
        return result.value

    async def write_tag(self, tag_name: str, value: Any, data_type: str) -> None:
        """Write a CIP tag."""
        result = self._driver.write_tag(tag_name, value, data_type)
        if result.error:
            raise RuntimeError(f"CIP write error: {result.error}")

    async def read_digital(self, address: IOAddress) -> bool:
        # Map to a tag name convention: "Local:{slot}:O.Data[{byte}].{bit}"
        tag = f"Local:{address.slot}:O.Data[{address.byte_offset}].{address.bit_offset}"
        val = await self.read_tag(tag)
        return bool(val)

    async def write_digital(self, address: IOAddress, value: bool) -> None:
        tag = f"Local:{address.slot}:O.Data[{address.byte_offset}].{address.bit_offset}"
        await self.write_tag(tag, value, "BOOL")


# ---------------------------------------------------------------------------
# PLC Program abstraction (IEC 61131-3 style)
# ---------------------------------------------------------------------------

class PLCProgram:
    """
    Abstract PLC program container.

    Mimics a Program Organisation Unit (POU) in IEC 61131-3.
    Contains a cyclic task that executes rungs and function blocks.
    """
    def __init__(self, name: str, cycle_time_ms: float = 10.0) -> None:
        self.name = name
        self.cycle_time_ms = cycle_time_ms
        self.rungs: List[Rung] = []
        self.function_blocks: List[FunctionBlock] = []
        self._running = False

    def add_rung(self, rung: Rung) -> None:
        self.rungs.append(rung)

    def add_fb(self, fb: FunctionBlock) -> None:
        self.function_blocks.append(fb)

    async def run_cycle(self) -> None:
        """Execute one PLC scan cycle."""
        for rung in self.rungs:
            rung.evaluate()
        for fb in self.function_blocks:
            if hasattr(fb, "execute"):
                fb.execute({})

    async def run(self) -> None:
        """Continuous cyclic execution."""
        self._running = True
        while self._running:
            t0 = time.monotonic()
            await self.run_cycle()
            elapsed = (time.monotonic() - t0) * 1000
            sleep_ms = max(0.0, self.cycle_time_ms - elapsed)
            await asyncio.sleep(sleep_ms / 1000.0)

    def stop(self) -> None:
        self._running = False
