"""
Siglent SDG2000X function generator driver (written for the SDG2082X).

Talks SCPI through pyvisa, over USB or LAN. Command syntax from Siglent's
"SDG Programming Guide":
    C1:BSWV WVTP,SINE          set one waveform parameter of channel 1
    C1:BSWV?                   -> C1:BSWV WVTP,SINE,FRQ,100HZ,PERI,0.01S,AMP,2V,OFST,0V,...
    C1:OUTP ON | OFF | LOAD,50 | LOAD,HZ
    C1:OUTP?                   -> C1:OUTP ON,LOAD,HZ,PLRT,NOR

The SDG2000X has no error query (no SYST:ERR?). So every setting is read back
and compared with what was asked; a mismatch raises instead of passing silently.

Usage
-----
    from labtools.sdg import SiglentSDG

    with SiglentSDG("TCPIP0::192.168.1.50::INSTR") as sdg:
        sdg.set_load(1, 50)                         # declare the real load FIRST
        sdg.set_waveform(1, "SINE", frequency_hz=1e3, amplitude_vpp=0.5)
        sdg.set_output(1, True)
        meta = sdg.get_metadata()                   # pass into provenance.stamp()

About amplitude and load
------------------------
AMP is the peak-to-peak voltage the generator produces ACROSS THE LOAD YOU
DECLARE with set_load(). If the real load differs, so does the real amplitude:
declared 50 ohm but driving a high-impedance input gives about twice the set
amplitude. (Standard function-generator behaviour; check against your scope.)
"""

from __future__ import annotations

import re
import time

import pyvisa

CHANNELS = (1, 2)
SHAPES = ("SINE", "SQUARE", "RAMP", "PULSE", "ARB")   # periodic shapes with a frequency
SHAPES_WITHOUT_PHASE = ("PULSE",)                     # the generator reports no PHSE for these
# Read-back check: (absolute, relative) tolerance per quantity. The generator
# rounds settings to its resolution and may report fewer digits, so an exact
# comparison would fail good settings. These are loose enough for rounding but
# far tighter than a clamped (out-of-range) value.
TOLERANCES = {
    "frequency_hz": (1e-6, 1e-4),
    "amplitude_vpp": (1e-3, 1e-3),     # ~1 mV resolution
    "offset_v": (1e-3, 1e-3),
    "phase_deg": (0.1, 0.0),
}

NUMBER = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

# The SDG drops commands that arrive too quickly one after another; a short
# pause after each write lets it process them. Read-backs still catch any loss.
PAUSE_AFTER_WRITE_S = 0.05


# --------------------------------------------------------------------------- #
# Parsing the generator's replies
# --------------------------------------------------------------------------- #

def to_number(text: str) -> float:
    """'100HZ' -> 100.0,  '-1V' -> -1.0,  '0.01S' -> 0.01,  '0' -> 0.0"""
    match = NUMBER.match(text.strip())
    if match is None:
        raise ValueError(f"No number in {text!r}")
    return float(match.group())


def reply_tokens(reply: str) -> list[str]:
    """Split 'C1:BSWV WVTP,SINE,FRQ,100HZ' into ['WVTP', 'SINE', 'FRQ', '100HZ'].
    Everything before the first space is the echoed command and is dropped."""
    if " " not in reply.strip():
        raise ValueError(f"Unexpected reply {reply!r}")
    _, values = reply.strip().split(" ", 1)
    return [token.strip() for token in values.split(",")]


def as_pairs(tokens: list[str]) -> dict[str, str]:
    """['WVTP', 'SINE', 'FRQ', '100HZ'] -> {'WVTP': 'SINE', 'FRQ': '100HZ'}"""
    pairs = {}
    for name, value in zip(tokens[0::2], tokens[1::2]):
        pairs[name] = value
    return pairs


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

class SiglentSDG:
    """Siglent SDG2000X function generator.

    Parameters
    ----------
    resource : VISA address, e.g. "TCPIP0::192.168.1.50::INSTR" or
               "USB0::0xF4EC::0x1102::SDG2XxxxxxxXXXX::INSTR"
    timeout_ms : how long to wait for a reply before raising
    outputs_off_on_close : switch both outputs off when closing. Off by default,
               because an experiment may need the signal to keep running.
    """

    def __init__(self, resource: str, timeout_ms: int = 5000,
                 outputs_off_on_close: bool = False):
        self._outputs_off_on_close = outputs_off_on_close
        self._manager = pyvisa.ResourceManager()
        self._instrument = self._manager.open_resource(resource)
        try:
            self._instrument.timeout = timeout_ms
            self._instrument.write_termination = "\n"
            self._instrument.read_termination = "\n"

            self.idn = self.query("*IDN?")
            if "SDG" not in self.idn:
                raise RuntimeError(f"{resource} is not a Siglent SDG: {self.idn!r}")
        except Exception:
            self._instrument.close()
            raise

    # ---- context manager ------------------------------------------------- #

    def __enter__(self) -> SiglentSDG:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        """Release the connection (and switch outputs off if asked to)."""
        if self._instrument is None:
            return
        try:
            if self._outputs_off_on_close:
                for channel in CHANNELS:
                    self.set_output(channel, False)
        finally:
            self._instrument.close()
            self._instrument = None

    # ---- raw communication ----------------------------------------------- #

    def write(self, command: str) -> None:
        self._instrument.write(command)
        time.sleep(PAUSE_AFTER_WRITE_S)          # see PAUSE_AFTER_WRITE_S

    def query(self, command: str) -> str:
        return self._instrument.query(command).strip()

    # ---- waveform -------------------------------------------------------- #

    def set_waveform(self, channel: int, shape: str, frequency_hz: float,
                     amplitude_vpp: float, offset_v: float = 0.0,
                     phase_deg: float = 0.0) -> None:
        """Set the basic waveform of one channel, then confirm it by read-back.

        Parameters are sent one per command: the shape first, because changing
        the shape can change the other parameters.
        """
        check_channel(channel)
        shape = shape.upper()
        if shape not in SHAPES:
            raise ValueError(f"shape must be one of {SHAPES}, got {shape!r}")
        has_phase = shape not in SHAPES_WITHOUT_PHASE
        if not has_phase and phase_deg != 0:
            raise ValueError(f"{shape} has no phase setting; shift it in time with its delay instead")

        self.write(f"C{channel}:BSWV WVTP,{shape}")
        self.write(f"C{channel}:BSWV FRQ,{frequency_hz:.12g}")
        self.write(f"C{channel}:BSWV AMP,{amplitude_vpp:.12g}")
        self.write(f"C{channel}:BSWV OFST,{offset_v:.12g}")
        if has_phase:
            self.write(f"C{channel}:BSWV PHSE,{phase_deg:.12g}")

        # No error query exists, so compare what the generator now reports
        # with what was asked. Out-of-range values are caught here.
        actual = self.waveform(channel)
        wanted = {
            "frequency_hz": frequency_hz,
            "amplitude_vpp": amplitude_vpp,
            "offset_v": offset_v,
            "phase_deg": phase_deg % 360,          # the generator wraps phase
        }
        if actual["shape"] != shape:
            raise RuntimeError(f"C{channel}: asked for {shape}, generator reports {actual['shape']}")
        if not has_phase:
            del wanted["phase_deg"]                 # nothing to compare
        for name, value in wanted.items():
            absolute, relative = TOLERANCES[name]
            if not close_enough(actual[name], value, absolute, relative):
                raise RuntimeError(f"C{channel}: asked for {name} = {value}, generator "
                                   f"reports {actual[name]} (out of range for this "
                                   f"shape, frequency or load?)")

    def waveform(self, channel: int) -> dict:
        """The basic waveform as the generator reports it, in SI units.
        phase_deg is None for shapes without a phase (e.g. PULSE)."""
        check_channel(channel)
        pairs = as_pairs(reply_tokens(self.query(f"C{channel}:BSWV?")))

        phase_deg = None
        if "PHSE" in pairs:
            phase_deg = to_number(pairs["PHSE"])

        return {
            "shape": pairs["WVTP"],
            "frequency_hz": to_number(pairs["FRQ"]),
            "amplitude_vpp": to_number(pairs["AMP"]),
            "offset_v": to_number(pairs["OFST"]),
            "phase_deg": phase_deg,
        }

    # ---- output ---------------------------------------------------------- #

    def set_output(self, channel: int, on: bool) -> None:
        """Switch one output on or off, and confirm."""
        check_channel(channel)
        self.write(f"C{channel}:OUTP {'ON' if on else 'OFF'}")
        if self.output(channel)["on"] != on:
            raise RuntimeError(f"C{channel}: output did not switch {'on' if on else 'off'}")

    def set_load(self, channel: int, load) -> None:
        """Declare the load the output drives: a resistance in ohms (50 ... 100000)
        or "HZ" for high impedance. Do this BEFORE setting the amplitude."""
        check_channel(channel)
        if isinstance(load, str):
            if load.upper() != "HZ":
                raise ValueError(f"load must be ohms or 'HZ', got {load!r}")
            self.write(f"C{channel}:OUTP LOAD,HZ")
        else:
            if not 50 <= load <= 100000:
                raise ValueError(f"load must be 50 to 100000 ohm or 'HZ', got {load}")
            self.write(f"C{channel}:OUTP LOAD,{load:g}")

        reported = self.output(channel)["load_ohm"]
        if isinstance(load, str):
            ok = reported == "HZ"
        else:
            ok = reported != "HZ" and close_enough(reported, load, absolute=0.5, relative=0.0)
        if not ok:
            raise RuntimeError(f"C{channel}: asked for load {load}, generator reports {reported}")

    def output(self, channel: int) -> dict:
        """Output state: on/off, declared load (ohms, or 'HZ'), polarity."""
        check_channel(channel)
        tokens = reply_tokens(self.query(f"C{channel}:OUTP?"))
        state = tokens[0]                     # 'ON' or 'OFF' comes first, alone
        pairs = as_pairs(tokens[1:])

        load = pairs.get("LOAD", "")
        load_ohm = "HZ" if load.upper() == "HZ" else to_number(load)
        return {
            "on": state.upper() == "ON",
            "load_ohm": load_ohm,
            "polarity": pairs.get("PLRT", ""),
        }

    # ---- synchronized start: camera trigger pulse + triangle sweep -------- #

    def setup_triggered_sweep(self, ramp_time_s: float, low_v: float, high_v: float,
                              after_top: str = "triangle",
                              pulse_vpp: float = 5.0, pulse_width_s: float = 100e-6,
                              start_phase_deg: float = 0.0,
                              pulse_channel: int = 1, sweep_channel: int = 2,
                              sleep_time_s: float = 0.5) -> None:
        """Arm one channel to fire ONE trigger pulse and the other to sweep ONCE
        from low_v up to high_v in exactly ramp_time_s, both started by the same
        trigger. Which channel does what is set by pulse_channel and
        sweep_channel (default CH1 pulse, CH2 sweep).

        Wiring:  pulse channel output --> camera Line0 (trigger input)

        Pulse channel: one pulse, 0 V -> pulse_vpp, burst of 1 cycle.
        Sweep channel: one cycle of a ramp whose rising part lasts ramp_time_s.
             after_top = "triangle": back down over another ramp_time_s
                         (50 % symmetry, period 2 x ramp_time_s, no voltage step)
             after_top = "sawtooth": straight back down at the top
                         (100 % symmetry, period ramp_time_s, abrupt step)

        start_phase_deg must put the start of the burst at the BOTTOM of the
        ramp. Where phase 0 lies on the generator's ramp is not documented:
        check on a scope once and set it accordingly.
        "COUP TRDUCH,ON" makes one trigger start both channels' bursts, so a
        single fire() starts the camera pulse and the sweep together.
        (TRDUCH is not in the SDG programming guides checked; it is confirmed
        here only by reading it back. Verify the two starts on a scope.)

        Fallback if TRDUCH is unavailable: cable CH1 also to the rear Aux In/Out,
        set CH2 to "BTWV TRSR,EXT" and "EDGE,RISE"; CH2 then starts on CH1's edge
        after a fixed external-trigger latency.

        Both outputs are switched on here, but nothing happens until fire().
        Declare the loads with set_load() first: the pulse amplitude assumes it.
        """
        check_channel(pulse_channel)
        check_channel(sweep_channel)
        if pulse_channel == sweep_channel:
            raise ValueError("pulse_channel and sweep_channel must differ")
        pulse = f"C{pulse_channel}"              # command prefix, e.g. "C1"
        sweep = f"C{sweep_channel}"

        # Pulse channel: single trigger pulse from 0 V up to pulse_vpp
        self.set_waveform(pulse_channel, "PULSE", frequency_hz=1 / (10 * pulse_width_s),
                          amplitude_vpp=pulse_vpp, offset_v=pulse_vpp / 2)
        self.write(f"{pulse}:BSWV WIDTH,{pulse_width_s:.12g}")
        self.write(f"{pulse}:BTWV STATE,ON")
        self.write(f"{pulse}:BTWV GATE_NCYC,NCYC")
        self.write(f"{pulse}:BTWV TRSR,MAN")     # started by fire()
        self.write(f"{pulse}:BTWV TIME,1")       # exactly one pulse

        # Sweep channel: one ramp from low_v to high_v, rising for exactly
        # ramp_time_s. Symmetry is the fraction of the period spent rising.
        if high_v <= low_v:
            raise ValueError(f"high_v ({high_v}) must be above low_v ({low_v})")
        if after_top == "triangle":
            symmetry_percent, period_s = 50, 2 * ramp_time_s
        elif after_top == "sawtooth":
            symmetry_percent, period_s = 100, ramp_time_s
        else:
            raise ValueError(f"after_top must be 'triangle' or 'sawtooth', got {after_top!r}")

        self.set_waveform(sweep_channel, "RAMP", 
                          frequency_hz=1 / period_s,
                          amplitude_vpp=high_v - low_v, 
                          offset_v=(high_v + low_v) / 2)
        self.write(f"{sweep}:BSWV SYM,{symmetry_percent}")
        self.write(f"{sweep}:BTWV STATE,ON")
        self.write(f"{sweep}:BTWV GATE_NCYC,NCYC")
        self.write(f"{sweep}:BTWV TRSR,MAN")
        self.write(f"{sweep}:BTWV TIME,1")       # a single sweep
        self.write(f"{sweep}:BTWV STPS,{start_phase_deg:.12g}")   # where in the cycle it starts

        # One trigger for both channels. Coupling settings can only be changed
        # while channel tracking is off.
        self.write("COUP TRACE,OFF")
        self.write("COUP TRDUCH,ON")

        # Confirm by read-back, as for everything else.
        self._check_burst(pulse_channel, {"STATE": "ON", "TRSR": "MAN"}, n_cycles=1)
        self._check_burst(sweep_channel, {"STATE": "ON", "TRSR": "MAN"}, n_cycles=1)
        symmetry = as_pairs(reply_tokens(self.query(f"{sweep}:BSWV?"))).get("SYM", "")
        if symmetry and to_number(symmetry) != symmetry_percent:
            raise RuntimeError(f"{sweep}: asked for {symmetry_percent} % symmetry, "
                               f"generator reports {symmetry}")
        coupling = self.query("COUP?")
        if as_pairs(reply_tokens(coupling)).get("TRDUCH", "").upper() != "ON":
            raise RuntimeError(f"Dual-channel trigger not confirmed; generator reports "
                               f"{coupling!r}. Use the cabled fallback (see docstring).")

        self.set_output(pulse_channel, True)
        self.set_output(sweep_channel, True)
        self._pulse_channel = pulse_channel      # remembered for fire()
        time.sleep(sleep_time_s)                  # let the generator settle before fire()

    def fire(self) -> None:
        """Trigger both bursts: the pulse and the sweep start together.
        Arm the camera BEFORE calling this, or the pulse is missed.
        
        WARNING: arm camera and scope trigger after changing the signal genreator 
                    settings otherwise chaning the settings may cause accidental 
                    triggering of the camera and scope.
        """
        pulse_channel = getattr(self, "_pulse_channel", None)
        if pulse_channel is None:
            raise RuntimeError("Call setup_triggered_sweep() before fire()")
        self.write(f"C{pulse_channel}:BTWV MTRIG")

    def _check_burst(self, channel: int, expected: dict, n_cycles: int) -> None:
        """Compare the burst settings the generator reports with what was set."""
        reply = self.query(f"C{channel}:BTWV?")
        tokens = reply_tokens(reply)
        # The reply ends with the carrier waveform after a "CARR" marker
        # (CARR,WVTP,PULSE,FRQ,...). Only the burst part is name,value pairs.
        if "CARR" in tokens:
            tokens = tokens[:tokens.index("CARR")]
        pairs = as_pairs(tokens)
        for name, value in expected.items():
            if pairs.get(name, "").upper() != value:
                raise RuntimeError(f"C{channel} burst: expected {name} = {value}, "
                                   f"generator reports {reply!r}")
        if "TIME" not in pairs or to_number(pairs["TIME"]) != n_cycles:
            raise RuntimeError(f"C{channel} burst: expected {n_cycles} cycle(s), "
                               f"generator reports {reply!r}")

    # ---- provenance ------------------------------------------------------ #

    def get_metadata(self) -> dict:
        """Identity and the state of both channels, for provenance.stamp()."""
        metadata = {"idn": self.idn}
        for channel in CHANNELS:
            metadata[f"channel_{channel}"] = {**self.waveform(channel), **self.output(channel)}
        return metadata


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def check_channel(channel: int) -> None:
    if channel not in CHANNELS:
        raise ValueError(f"channel must be 1 or 2, got {channel!r}")


def close_enough(actual: float, wanted: float, absolute: float, relative: float) -> bool:
    """True if actual is within absolute + relative * |wanted| of wanted."""
    return abs(actual - wanted) <= absolute + relative * abs(wanted)


if __name__ == "__main__":
    # Bench self-test:  python -m labtools.sdg "TCPIP0::192.168.1.50::INSTR"
    # Changes channel 1 settings; its output stays off.
    import sys

    with SiglentSDG(sys.argv[1]) as sdg:
        print(sdg.idn)
        sdg.set_output(1, False)
        sdg.set_load(1, "HZ")
        sdg.set_waveform(1, "SINE", frequency_hz=1000, amplitude_vpp=0.1)
        print(sdg.get_metadata())