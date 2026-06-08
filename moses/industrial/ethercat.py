"""
moses/industrial/ethercat.py
============================

EtherCAT master/slave integration for humanoid robot servo control.

Implements:
  - EtherCAT frame structure (ETG.1000.1, ETG.1000.2)
  - Process Data Object (PDO) mapping for servo drives
  - Distributed Clocks (DC) synchronization
  - CiA 402 drive profile (CANopen over EtherCAT, CoE)

Standards:
  - IEC 61158 (EtherCAT protocol)
  - ETG.1000.1 - EtherCAT Technology Group specification
  - ETG.1000.2 - EtherCAT Slave Information (ESI)
  - CiA 402 (IEC 61800-7-201) - Device profile for drives
  - IEC 61800-5-2 - Adjustable speed electrical power drive systems

Dependencies:
  - pysoem (SOEM Python bindings) for real EtherCAT master
  - Mock interfaces provided for simulation/testing

Author: Moses Industrial Team
Version: 6.0.0
"""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union


# ---------------------------------------------------------------------------
# EtherCAT constants and frame structure
# ---------------------------------------------------------------------------

class EtherCATCommand(IntEnum):
    """EtherCAT datagram commands (ETG.1000.1, Section 5.2)."""
    NOP = 0x00          # No Operation
    APRD = 0x01         # Auto Increment Physical Read
    APWR = 0x02         # Auto Increment Physical Write
    APRW = 0x03         # Auto Increment Physical ReadWrite
    FPRD = 0x04         # Configured Address Physical Read
    FPWR = 0x05         # Configured Address Physical Write
    FPRW = 0x06         # Configured Address Physical ReadWrite
    BRD = 0x07          # Broadcast Read
    BWR = 0x08          # Broadcast Write
    BRW = 0x09          # Broadcast ReadWrite
    LRD = 0x0A          # Logical Read
    LWR = 0x0B          # Logical Write
    LRW = 0x0C          # Logical ReadWrite
    ARMW = 0x0D         # Auto Increment Physical Read Multiple Write
    FRMW = 0x0E         # Configured Address Physical Read Multiple Write


class EtherCATState(IntEnum):
    """EtherCAT device states (ETG.1000.1, Section 6.1)."""
    INIT = 0x01
    PRE_OP = 0x02
    BOOT = 0x03
    SAFE_OP = 0x04
    OPERATIONAL = 0x08


@dataclass(frozen=True)
class EtherCATDatagram:
    """
    EtherCAT datagram structure.

    Frame format (ETG.1000.1):
      - Command (1 byte)
      - Index (1 byte)
      - Address (4 bytes: ADP 2 + ADO 2)
      - Length (2 bytes, bit 15 = CIRCUIT_BREAK)
      - Interrupt (2 bytes)
      - Data (0..1486 bytes)
      - Working Counter (2 bytes)
    """
    command: EtherCATCommand
    index: int = 0
    adp: int = 0x0000       # Auto-increment / configured address
    ado: int = 0x0000       # Physical/logical address offset
    length: int = 0
    irq: int = 0x0000
    data: bytes = b""
    wkc: int = 0x0000       # Working Counter (appended by slaves)

    def pack(self) -> bytes:
        """Pack datagram into wire format."""
        header = struct.pack(
            "<BBHHHH",
            self.command,
            self.index,
            self.adp,
            self.ado,
            self.length | 0x0000,  # CIRCUIT_BREAK = 0
            self.irq,
        )
        return header + self.data + struct.pack("<H", self.wkc)

    @classmethod
    def unpack(cls, raw: bytes) -> EtherCATDatagram:
        """Unpack datagram from wire format."""
        cmd, idx, adp, ado, length, irq = struct.unpack("<BBHHHH", raw[:10])
        data_len = length & 0x7FFF
        data = raw[10:10 + data_len]
        wkc = struct.unpack("<H", raw[10 + data_len:12 + data_len])[0]
        return cls(
            command=EtherCATCommand(cmd),
            index=idx,
            adp=adp,
            ado=ado,
            length=data_len,
            irq=irq,
            data=data,
            wkc=wkc,
        )


@dataclass(frozen=True)
class EtherCATFrame:
    """
    EtherCAT frame (ETG.1000.1, Section 5.1).

    Encapsulated in Ethernet frame with EtherType 0x88A4.
    Can contain multiple datagrams.
    """
    datagrams: Tuple[EtherCATDatagram, ...] = ()

    def pack(self) -> bytes:
        """Pack frame with header and datagrams."""
        # EtherCAT header: length (11 bits) + reserved (1) + type (4)
        # Type 0x01 = EtherCAT datagrams
        total_len = sum(len(dg.pack()) for dg in self.datagrams)
        header = struct.pack("<H", (total_len & 0x07FF) | 0x1000)
        payload = b"".join(dg.pack() for dg in self.datagrams)
        return header + payload

    @classmethod
    def unpack(cls, raw: bytes) -> EtherCATFrame:
        """Unpack frame."""
        # Simplified: assumes single datagram for clarity
        header_len = struct.unpack("<H", raw[:2])[0] & 0x07FF
        offset = 2
        dgs: List[EtherCATDatagram] = []
        while offset < 2 + header_len:
            dg = EtherCATDatagram.unpack(raw[offset:])
            dgs.append(dg)
            offset += 10 + len(dg.data) + 2
        return cls(datagrams=tuple(dgs))


# ---------------------------------------------------------------------------
# PDO Mapping
# ---------------------------------------------------------------------------

class PDOType(Enum):
    """Process Data Object direction."""
    TX_PDO = auto()     # Slave -> Master (inputs)
    RX_PDO = auto()     # Master -> Slave (outputs)


@dataclass
class PDOEntry:
    """
    Single PDO mapping entry.

    Maps an object dictionary entry to a PDO.
    CiA 402 objects use 0x6040 (controlword), 0x6041 (statusword), etc.
    """
    index: int          # Object dictionary index (e.g., 0x6040)
    subindex: int       # Sub-index (e.g., 0x00)
    bit_length: int     # Size in bits (8, 16, 32)
    name: str = ""
    scale: float = 1.0
    offset: float = 0.0

    @property
    def byte_length(self) -> int:
        return (self.bit_length + 7) // 8


@dataclass
class PDORemapping:
    """
    PDO remapping configuration for a slave.

    Allows dynamic reconfiguration of PDO contents.
    Standard PDO assignments (CiA 402):
      - 0x1A00-0x1A03: TX-PDO mapping parameters
      - 0x1600-0x1603: RX-PDO mapping parameters
    """
    slave_id: int
    pdo_type: PDOType
    mapping_index: int    # 0x1600..0x1603 or 0x1A00..0x1A03
    entries: List[PDOEntry] = field(default_factory=list)

    def calculate_size(self) -> int:
        """Calculate total PDO size in bytes."""
        return sum(e.byte_length for e in self.entries)

    def pack(self, values: Dict[str, Union[int, float]]) -> bytes:
        """Pack dictionary of named values into PDO bytes."""
        buf = bytearray(self.calculate_size())
        offset = 0
        for entry in self.entries:
            val = values.get(entry.name, 0)
            if entry.bit_length == 8:
                struct.pack_into("<B", buf, offset, int(val))
            elif entry.bit_length == 16:
                struct.pack_into("<h", buf, offset, int(val))
            elif entry.bit_length == 32:
                if isinstance(val, float):
                    struct.pack_into("<f", buf, offset, val)
                else:
                    struct.pack_into("<i", buf, offset, int(val))
            offset += entry.byte_length
        return bytes(buf)

    def unpack(self, data: bytes) -> Dict[str, Union[int, float]]:
        """Unpack PDO bytes into dictionary of named values."""
        values: Dict[str, Union[int, float]] = {}
        offset = 0
        for entry in self.entries:
            if entry.bit_length == 8:
                values[entry.name] = struct.unpack_from("<B", data, offset)[0]
            elif entry.bit_length == 16:
                values[entry.name] = struct.unpack_from("<h", data, offset)[0]
            elif entry.bit_length == 32:
                if entry.scale != 1.0 or entry.offset != 0.0:
                    raw = struct.unpack_from("<i", data, offset)[0]
                    values[entry.name] = raw * entry.scale + entry.offset
                else:
                    values[entry.name] = struct.unpack_from("<i", data, offset)[0]
            offset += entry.byte_length
        return values


# Standard CiA 402 PDO mapping for servo drive
DEFAULT_TX_PDO = PDORemapping(
    slave_id=0,
    pdo_type=PDOType.TX_PDO,
    mapping_index=0x1A00,
    entries=[
        PDOEntry(0x6041, 0x00, 16, "statusword"),           # Statusword
        PDOEntry(0x6064, 0x00, 32, "position_actual"),      # Position actual value
        PDOEntry(0x606C, 0x00, 32, "velocity_actual"),      # Velocity actual value
        PDOEntry(0x6077, 0x00, 16, "torque_actual"),        # Torque actual value
        PDOEntry(0x603F, 0x00, 16, "error_code"),           # Error code
    ],
)

DEFAULT_RX_PDO = PDORemapping(
    slave_id=0,
    pdo_type=PDOType.RX_PDO,
    mapping_index=0x1600,
    entries=[
        PDOEntry(0x6040, 0x00, 16, "controlword"),          # Controlword
        PDOEntry(0x607A, 0x00, 32, "target_position"),      # Target position
        PDOEntry(0x60FF, 0x00, 32, "target_velocity"),      # Target velocity
        PDOEntry(0x6071, 0x00, 16, "target_torque"),        # Target torque
        PDOEntry(0x6060, 0x00, 8,  "modes_of_operation"),   # Modes of operation
    ],
)


# ---------------------------------------------------------------------------
# Distributed Clocks (DC)
# ---------------------------------------------------------------------------

@dataclass
class DistributedClocks:
    """
    EtherCAT Distributed Clocks synchronization.

    Provides sub-microsecond synchronization across all slaves.
    Reference clock is typically the first slave with DC capability.

    Key registers (ETG.1000.1):
      - 0x0900: ESC DL Control (activate DC)
      - 0x0910: DC Receive Time (port 0)
      - 0x0920: DC System Time
      - 0x0928: DC System Time Offset
      - 0x092C: DC System Time Delay
      - 0x0981: DC Activation Register
      - 0x09A0: DC Sync0 Cycle Time
    """
    cycle_time_ns: int = 1_000_000      # 1 ms default
    shift_time_ns: int = 0
    reference_slave: int = 0
    _system_time_offset: int = 0
    _propagation_delay: int = 0

    def calculate_offset(self, local_time: int, reference_time: int) -> int:
        """Calculate offset between local and reference clock."""
        self._system_time_offset = reference_time - local_time
        return self._system_time_offset

    def calculate_propagation_delay(self, delays: List[int]) -> int:
        """
        Calculate propagation delay through slave chain.

        Uses drift-compensated delay measurement per ETG.1000.1.
        """
        if not delays:
            self._propagation_delay = 0
            return 0
        # Average of forward and backward measurements
        self._propagation_delay = sum(delays) // len(delays)
        return self._propagation_delay

    def get_sync0_time(self, cycle_counter: int) -> int:
        """Calculate next SYNC0 event time in nanoseconds."""
        base = self._system_time_offset + self._propagation_delay
        return base + cycle_counter * self.cycle_time_ns + self.shift_time_ns

    def configure_sync0(
        self,
        activate: bool = True,
        cycle_time_ns: Optional[int] = None,
    ) -> bytes:
        """Generate register write data for DC Sync0 configuration."""
        if cycle_time_ns is not None:
            self.cycle_time_ns = cycle_time_ns
        reg_981 = 0x03 if activate else 0x00  # SYNC0 + SYNC1
        reg_9A0 = struct.pack("<Q", self.cycle_time_ns)
        return struct.pack("<B", reg_981) + reg_9A0


# ---------------------------------------------------------------------------
# CiA 402 Drive Profile
# ---------------------------------------------------------------------------

class CiA402State(IntEnum):
    """CiA 402 drive state machine (IEC 61800-7-201)."""
    NOT_READY_TO_SWITCH_ON = 0b0000_0000_0000_0000
    SWITCH_ON_DISABLED = 0b0100_0000_0000_0000
    READY_TO_SWITCH_ON = 0b0010_0001_0000_0000
    SWITCHED_ON = 0b0010_0011_0000_0000
    OPERATION_ENABLED = 0b0010_0111_0000_0000
    QUICK_STOP_ACTIVE = 0b0000_0111_0000_0000
    FAULT_REACTION_ACTIVE = 0b0000_1111_0000_0000
    FAULT = 0b0000_1000_0000_0000


class CiA402ControlWord(IntEnum):
    """Controlword bit definitions (CiA 402, 0x6040)."""
    SWITCH_ON = 0x0001
    ENABLE_VOLTAGE = 0x0002
    QUICK_STOP = 0x0004
    ENABLE_OPERATION = 0x0008
    FAULT_RESET = 0x0080
    HALT = 0x0100


class CiA402ModeOfOperation(IntEnum):
    """Modes of operation (CiA 402, 0x6060)."""
    PROFILE_POSITION = 0x01
    PROFILE_VELOCITY = 0x03
    PROFILE_TORQUE = 0x04
    HOMING = 0x06
    INTERPOLATED_POSITION = 0x07
    CYCLIC_SYNC_POSITION = 0x08
    CYCLIC_SYNC_VELOCITY = 0x09
    CYCLIC_SYNC_TORQUE = 0x0A


@dataclass
class CiA402DriveProfile:
    """
    CiA 402 (IEC 61800-7-201) drive profile implementation.

    Manages the drive state machine and PDO mapping for servo drives.
    Supports cyclic synchronous position/velocity/torque modes.
    """
    slave_id: int
    tx_pdo: PDORemapping = field(default_factory=lambda: DEFAULT_TX_PDO)
    rx_pdo: PDORemapping = field(default_factory=lambda: DEFAULT_RX_PDO)
    _state: CiA402State = CiA402State.NOT_READY_TO_SWITCH_ON
    _mode: CiA402ModeOfOperation = CiA402ModeOfOperation.CYCLIC_SYNC_POSITION

    def parse_statusword(self, statusword: int) -> CiA402State:
        """Extract state from statusword (0x6041)."""
        # Mask bits 0-3 and 5-6
        masked = statusword & 0b0100_0111_0000_1111
        for state in CiA402State:
            if state.value == masked:
                self._state = state
                return state
        return self._state

    def build_controlword(self, target_state: CiA402State) -> int:
        """
        Build controlword to transition to target state.

        State transitions (CiA 402, Figure 25):
          Shutdown:          0x0006 (Switch On Disabled -> Ready)
          Switch On:         0x0007 (Ready -> Switched On)
          Enable Operation:  0x000F (Switched On -> Operation Enabled)
          Disable Operation: 0x0007 (Operation -> Switched On)
          Disable Voltage:   0x0000 (any -> Switch On Disabled)
          Quick Stop:        0x0002 (any -> Quick Stop)
          Fault Reset:       0x0080 (Fault -> Switch On Disabled)
        """
        transitions = {
            CiA402State.READY_TO_SWITCH_ON: 0x0006,
            CiA402State.SWITCHED_ON: 0x0007,
            CiA402State.OPERATION_ENABLED: 0x000F,
            CiA402State.SWITCH_ON_DISABLED: 0x0000,
            CiA402State.QUICK_STOP_ACTIVE: 0x0002,
        }
        return transitions.get(target_state, 0x0000)

    def enable_drive(self) -> int:
        """Return controlword sequence to enable operation."""
        return self.build_controlword(CiA402State.OPERATION_ENABLED)

    def disable_drive(self) -> int:
        """Return controlword to disable voltage."""
        return self.build_controlword(CiA402State.SWITCH_ON_DISABLED)

    def fault_reset(self) -> int:
        """Return controlword to reset fault."""
        return CiA402ControlWord.FAULT_RESET.value

    def set_mode(self, mode: CiA402ModeOfOperation) -> int:
        """Set mode of operation."""
        self._mode = mode
        return mode.value

    def create_rx_pdo_data(
        self,
        target_position: int = 0,
        target_velocity: int = 0,
        target_torque: int = 0,
        controlword: Optional[int] = None,
    ) -> bytes:
        """Pack RX-PDO data for cyclic operation."""
        cw = controlword if controlword is not None else self.enable_drive()
        return self.rx_pdo.pack({
            "controlword": cw,
            "target_position": target_position,
            "target_velocity": target_velocity,
            "target_torque": target_torque,
            "modes_of_operation": self._mode.value,
        })

    def parse_tx_pdo_data(self, data: bytes) -> Dict[str, Union[int, float]]:
        """Unpack TX-PDO data from slave."""
        values = self.tx_pdo.unpack(data)
        if "statusword" in values:
            self.parse_statusword(int(values["statusword"]))
        return values


# ---------------------------------------------------------------------------
# EtherCAT Master
# ---------------------------------------------------------------------------

class EtherCATMaster:
    """
    EtherCAT master controller.

    Wraps SOEM (Simple Open EtherCAT Master) via pysoem.
    Manages slave enumeration, state transitions, and cyclic process data.
    """
    def __init__(self, adapter_name: str = "eth0", cycle_time_us: int = 1000) -> None:
        self.adapter_name = adapter_name
        self.cycle_time_us = cycle_time_us
        self._master: Any = None
        self.slaves: List[EtherCATSlave] = []
        self.dc: Optional[DistributedClocks] = None
        self._running = False

    def open(self) -> None:
        """Initialize EtherCAT master on network adapter."""
        try:
            import pysoem
            self._master = pysoem.Master()
            self._master.open(self.adapter_name)
        except ImportError:
            raise RuntimeError("pysoem not installed. Install: pip install pysoem")

    def close(self) -> None:
        if self._master:
            self._master.close()
            self._master = None

    def scan_slaves(self) -> int:
        """Scan for slaves and return count."""
        if not self._master:
            raise RuntimeError("Master not opened")
        slave_count = self._master.config_init()
        self.slaves = [
            EtherCATSlave(
                slave_id=i,
                name=self._master.slaves[i].name,
                config_addr=self._master.slaves[i].configadr,
            )
            for i in range(1, slave_count + 1)
        ]
        return slave_count

    def configure_dc(self, reference_slave: int = 1) -> DistributedClocks:
        """Configure Distributed Clocks with first DC-capable slave as reference."""
        self.dc = DistributedClocks(
            cycle_time_ns=self.cycle_time_us * 1000,
            reference_slave=reference_slave,
        )
        if self._master:
            self._master.config_dc()
        return self.dc

    def transition_state(self, target_state: EtherCATState) -> None:
        """Transition all slaves to target state."""
        if not self._master:
            return
        state_map = {
            EtherCATState.INIT: 1,
            EtherCATState.PRE_OP: 2,
            EtherCATState.SAFE_OP: 4,
            EtherCATState.OPERATIONAL: 8,
        }
        self._master.state_check(0, state_map[target_state], 5000)

    def send_process_data(self) -> int:
        """Send/receive one cycle of process data. Returns working counter."""
        if self._master:
            return self._master.send_processdata()
        return 0

    def receive_process_data(self, timeout_us: int = 2000) -> int:
        """Receive process data. Returns working counter."""
        if self._master:
            return self._master.receive_processdata(timeout_us)
        return 0

    def run_cyclic(self, callback: Callable[[], None]) -> None:
        """Run cyclic process data exchange."""
        self._running = True
        while self._running:
            self.send_process_data()
            self.receive_process_data()
            callback()
            time.sleep(self.cycle_time_us / 1_000_000.0)

    def stop(self) -> None:
        self._running = False


@dataclass
class EtherCATSlave:
    """EtherCAT slave device descriptor."""
    slave_id: int
    name: str
    config_addr: int
    state: EtherCATState = EtherCATState.INIT
    tx_pdo: Optional[PDORemapping] = None
    rx_pdo: Optional[PDORemapping] = None
    drive_profile: Optional[CiA402DriveProfile] = None
    has_dc: bool = False

    def configure_pdo(self, tx: PDORemapping, rx: PDORemapping) -> None:
        self.tx_pdo = tx
        self.rx_pdo = rx

    def configure_cia402(self) -> CiA402DriveProfile:
        """Configure CiA 402 drive profile for this slave."""
        self.drive_profile = CiA402DriveProfile(
            slave_id=self.slave_id,
            tx_pdo=self.tx_pdo or DEFAULT_TX_PDO,
            rx_pdo=self.rx_pdo or DEFAULT_RX_PDO,
        )
        return self.drive_profile


# ---------------------------------------------------------------------------
# Mock / Simulation interfaces
# ---------------------------------------------------------------------------

class MockEtherCATMaster(EtherCATMaster):
    """Mock EtherCAT master for simulation and testing."""
    def __init__(self, adapter_name: str = "mock0", cycle_time_us: int = 1000) -> None:
        super().__init__(adapter_name, cycle_time_us)
        self._mock_slaves: List[EtherCATSlave] = []
        self._pd_buffer: Dict[int, bytes] = {}

    def open(self) -> None:
        pass

    def close(self) -> None:
        pass

    def scan_slaves(self) -> int:
        # Simulate 3 slaves: 2 servo drives + 1 I/O module
        self._mock_slaves = [
            EtherCATSlave(1, "Maxon EPOS4", 0x1001, has_dc=True),
            EtherCATSlave(2, "Beckhoff EL7201", 0x1002, has_dc=True),
            EtherCATSlave(3, "Beckhoff EL2008", 0x1003, has_dc=False),
        ]
        self.slaves = self._mock_slaves
        return len(self._mock_slaves)

    def send_process_data(self) -> int:
        return len(self._mock_slaves)

    def receive_process_data(self, timeout_us: int = 2000) -> int:
        # Simulate TX-PDO data
        for slave in self._mock_slaves:
            if slave.drive_profile:
                self._pd_buffer[slave.slave_id] = slave.drive_profile.tx_pdo.pack({
                    "statusword": 0x0237,      # Operation enabled
                    "position_actual": 10000,
                    "velocity_actual": 500,
                    "torque_actual": 100,
                    "error_code": 0,
                })
        return len(self._mock_slaves)

    def read_pdo(self, slave_id: int) -> Dict[str, Union[int, float]]:
        """Read parsed PDO data from mock buffer."""
        slave = next((s for s in self._mock_slaves if s.slave_id == slave_id), None)
        if slave and slave.drive_profile and slave_id in self._pd_buffer:
            return slave.drive_profile.parse_tx_pdo_data(self._pd_buffer[slave_id])
        return {}

    def write_pdo(self, slave_id: int, data: bytes) -> None:
        self._pd_buffer[slave_id] = data
