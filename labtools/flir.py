"""
FLIR (Spinnaker / PySpin) camera driver.

Author: Mykhailo Vorobiov

Conventions
-----------
- Frames are uint16 arrays indexed (y, x), i.e. (row, column), the camera's native
  order; sequences are (frame, y, x). imshow(frame) displays correctly, and
  frame[iy, ix] pairs with (y_mm[iy], x_mm[ix]).
  Convert to float for any arithmetic, e.g. frame.astype(float) or frame / 65535.0.
- Timestamps come from the camera's own clock (chunk data), in ns internally;
  sequences return them relative to the first frame, in microseconds.
- Errors are raised, never printed and swallowed.

Usage
-----
    from labtools.flir import FLIRCamera

    with FLIRCamera(exposure_value=20000, gain_value=10) as cam:
        frame = cam.get_frame()
        seq = cam.get_frame_sequence(100)
        meta = cam.get_metadata()       # pass straight into provenance.stamp()

Triggered acquisition (pulse on Line0 starts the stream)
    cam.trigger("ON")
    cam.arm()                               # camera now waits for the edge
    sigen.fire()                            # the pulse
    seq = cam.collect(200)
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

import PySpin


class IncompleteImage(RuntimeError):
    """A frame arrived damaged, e.g. with missing network packets (GigE).
    Its own type, so callers can replace such a frame without hiding other errors."""


def describe_image_status(status: int) -> str:
    """Spinnaker's text for an image status code, falling back to the number."""
    try:
        return f"{PySpin.Image.GetImageStatusDescription(status)} (status {status})"
    except Exception:
        return f"status {status}"


def warn_if_gaps(time_us: NDArray) -> None:
    """Warn if any interval between frames is much longer than the typical one.

    A frame lost in transit (or replaced after damage) never shows up as an
    error by itself: the only trace is a longer gap in the camera timestamps.
    For a synchronized sweep that gap shifts every later frame's sweep value.
    """
    if len(time_us) < 3:
        return
    intervals = np.diff(time_us)
    typical = np.median(intervals)
    long_gaps = np.nonzero(intervals > 1.5 * typical)[0]
    if len(long_gaps) > 0:
        print(f"warning: {len(long_gaps)} gap(s) longer than 1.5 x the typical "
              f"{typical:.0f} us frame interval, after frames {long_gaps.tolist()}")


def warn_if_large(num_frames: int, width: int, height: int) -> None:
    """Warn if the frame array for a sequence would exceed MEMORY_WARNING_GB.

    The whole sequence is held in memory as uint16 (2 bytes per pixel), so a
    long full-frame run can exhaust RAM or push the computer into swapping.
    """
    size_gb = num_frames * width * height * 2 / 1e9
    if size_gb > MEMORY_WARNING_GB:
        print(f"warning: {num_frames} frames of {width} x {height} need "
              f"{size_gb:.1f} GB of memory (warning limit {MEMORY_WARNING_GB} GB)")


BLACK_LEVEL_PERCENT = 2.0     # fixed offset; high enough that dark noise is not clipped at 0
MEMORY_WARNING_GB = 1.0       # warn when a sequence needs more memory than this

# Frames the driver can hold between arm() and collect(). Each 720 x 540 Mono16
# frame takes 0.78 MB, so 100 buffers are about 80 MB of host memory.
STREAM_BUFFERS = 100

class FLIRCamera:
    """FLIR camera (Spinnaker / PySpin) for single-frame and sequence acquisition.

    Parameters
    ----------
    cam_index : int
        Index in the Spinnaker camera list.
    fov_horizontal : float
        Horizontal field of view in mm (sets the x_mm / y_mm axes; square pixels assumed).
    exposure_value : float
        Exposure time in microseconds.
    gain_value : float
        Analog gain in dB.
    """

    def __init__(self,
                 cam_index: int = 0,
                 fov_horizontal: float = 8.1,
                 exposure_value: float = 40000.0,
                 gain_value: float = 20.0):
        self._cam_index = cam_index
        self._fov_horizontal = fov_horizontal
        self._trigger_wait_ms = 0
        self._temperature_before = float("nan")   # set by arm(), reported by collect()
        self._cam = None
        self._cam_list = None

        self._system = PySpin.System.GetInstance()
        try:
            self._cam_list = self._system.GetCameras()
            n = self._cam_list.GetSize()
            if n == 0:
                raise RuntimeError("No FLIR cameras detected")
            if not 0 <= cam_index < n:
                raise IndexError(f"cam_index={cam_index}, but only {n} camera(s) detected")
            self._open(cam_index)
            self.set_exposure_time(exposure_value)
            self.set_gain(gain_value)
        except Exception:
            self._release()       # never leave the camera or the System half-open
            raise

    # ------------------------------------------------------------------ #
    # context manager
    # ------------------------------------------------------------------ #

    def __enter__(self) -> FLIRCamera:
        return self

    def __exit__(self, *exc) -> None:
        self.deinit()

    # ------------------------------------------------------------------ #
    # setup
    # ------------------------------------------------------------------ #

    def _open(self, cam_index: int) -> None:
        self._cam = self._cam_list.GetByIndex(cam_index)
        self._cam.Init()
        self._cam_index = cam_index
        self._configure()

    def _configure(self) -> None:
        """Put the camera in a known state. Required settings raise on failure."""
        cam = self._cam
        cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)
        cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
        cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
        cam.GainAuto.SetValue(PySpin.GainAuto_Off)
        if PySpin.IsWritable(cam.GammaEnable):        # not present on every model
            cam.GammaEnable.SetValue(False)
        self.set_px_format_mono16bit()
        cam.ChunkModeActive.SetValue(True)
        cam.ChunkSelector.SetValue(PySpin.ChunkSelector_Timestamp)
        cam.ChunkEnable.SetValue(True)
        self.set_fixed_black_level(BLACK_LEVEL_PERCENT)
        self.set_stream_buffers(STREAM_BUFFERS)

    def set_stream_buffers(self, count: int) -> None:
        """Keep every frame, oldest first, in `count` host buffers.

        Frames that arrive between arm() and collect() wait here instead of
        being thrown away. Only valid while not streaming.
        """
        stream = self._cam.TLStream
        stream.StreamBufferHandlingMode.SetValue(PySpin.StreamBufferHandlingMode_OldestFirst)
        stream.StreamBufferCountMode.SetValue(PySpin.StreamBufferCountMode_Manual)

        largest = stream.StreamBufferCountManual.GetMax()
        if count > largest:
            print(f"warning: {count} stream buffers requested, camera allows {largest}")
            count = largest
        stream.StreamBufferCountManual.SetValue(count)

    def set_fixed_black_level(self, percent: float) -> None:
        """Turn off automatic black-level clamping and set a fixed offset.

        With clamping on, the camera keeps re-adjusting the offset from its
        optically black pixels. Near a decision boundary it toggles between two
        settings, which shifts the whole frame by a fixed step at random times.
        A fixed offset is stable frame to frame; drift with temperature is
        handled by taking darks under the same conditions as the data.
        """
        cam = self._cam

        # Clamping first: while it is on, it may override the black level.
        if PySpin.IsWritable(cam.BlackLevelClampingEnable):
            cam.BlackLevelClampingEnable.SetValue(False)
        else:
            print("warning: black-level clamping cannot be switched off on this camera")

        # Only the total ("All") black level is user-writable.
        if PySpin.IsWritable(cam.BlackLevelSelector):
            cam.BlackLevelSelector.SetValue(PySpin.BlackLevelSelector_All)
        cam.BlackLevel.SetValue(percent)

    def set_px_format_mono16bit(self) -> None:
        """12-bit ADC, Mono16 pixel format. Only valid while not streaming."""
        self._cam.AdcBitDepth.SetValue(PySpin.AdcBitDepth_Bit12)
        self._cam.PixelFormat.SetValue(PySpin.PixelFormat_Mono16)

    def set_exposure_time(self, time_us: float) -> None:
        """Exposure in microseconds. The camera quantizes it; read back `exposure_value`."""
        node = self._cam.ExposureTime
        lo, hi = node.GetMin(), node.GetMax()
        if not lo <= time_us <= hi:
            raise ValueError(f"exposure {time_us} us outside camera range "
                             f"[{lo:.1f}, {hi:.1f}] us (max depends on frame-rate limit)")
        node.SetValue(time_us)

    def set_gain(self, gain_db: float) -> None:
        """Analog gain in dB, checked against the camera's own limits."""
        node = self._cam.Gain
        lo, hi = node.GetMin(), node.GetMax()
        if not lo <= gain_db <= hi:
            raise ValueError(f"gain {gain_db} dB outside camera range [{lo:.1f}, {hi:.1f}] dB")
        node.SetValue(gain_db)

    def trigger(self, on_off: Literal["ON", "OFF"] = "ON", wait_ms: int = 10_000) -> None:
        """Hardware trigger on Line0, rising edge, starting acquisition.

        wait_ms : how long a grab waits for the trigger before raising a timeout.
        """
        self._end()
        cam = self._cam
        cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)  # selector/source are locked while On
        if on_off == "ON":
            cam.TriggerSelector.SetValue(PySpin.TriggerSelector_AcquisitionStart)
            cam.TriggerSource.SetValue(PySpin.TriggerSource_Line0)
            cam.TriggerActivation.SetValue(PySpin.TriggerActivation_RisingEdge)
            cam.TriggerMode.SetValue(PySpin.TriggerMode_On)
            self._trigger_wait_ms = wait_ms
        elif on_off == "OFF":
            cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
            self._trigger_wait_ms = 0
        else:
            raise ValueError(f"on_off must be 'ON' or 'OFF', got {on_off!r}")

    def disable_fps_limit(self) -> None:
        for name in ("AcquisitionFrameRateEnable", "AcquisitionFrameRateEnabled"):
            node = getattr(self._cam, name, None)
            if node is not None and PySpin.IsWritable(node):
                node.SetValue(False)
                return
        raise RuntimeError("No writable frame-rate-enable node on this camera")

    def switch(self, cam_index: int) -> None:
        """Close the current camera and open another one, keeping exposure and gain."""
        if cam_index == self._cam_index:
            return
        exposure, gain = self.exposure_value, self.gain_value
        self._close_camera()
        self._open(cam_index)
        self.set_exposure_time(exposure)
        self.set_gain(gain)

    # ------------------------------------------------------------------ #
    # properties (read live from the camera, so they can't go stale)
    # ------------------------------------------------------------------ #

    @property
    def index(self) -> int:
        return self._cam_index

    @property
    def resolution(self) -> tuple[int, int]:
        """(width, height) in pixels, reflecting the current ROI."""
        return self._cam.Width.GetValue(), self._cam.Height.GetValue()

    @property
    def fov_horizontal(self) -> float:
        return self._fov_horizontal

    @property
    def pixel_pitch_mm(self) -> float:
        return self._fov_horizontal / self.resolution[0]

    @property
    def exposure_value(self) -> float:
        """Actual exposure in microseconds, as set on the camera."""
        return self._cam.ExposureTime.GetValue()

    @property
    def exposure_range_us(self) -> tuple[float, float]:
        """(min, max) exposure the camera allows right now, in microseconds.
        The max depends on the frame-rate limit; see disable_fps_limit()."""
        node = self._cam.ExposureTime
        return node.GetMin(), node.GetMax()

    @property
    def gain_value(self) -> float:
        """Actual gain in dB, as set on the camera."""
        return self._cam.Gain.GetValue()

    @property
    def temperature_c(self) -> float:
        """Device temperature in degrees C, or None if the camera does not report one.

        Read from the GenICam node DeviceTemperature. Which sensor it measures
        (sensor die or main board) depends on the model; check SpinView. Dark
        current roughly doubles every 6-8 C, so record it with every dark set.
        """
        node = getattr(self._cam, "DeviceTemperature", None)
        if node is None or not PySpin.IsReadable(node):
            return float("nan")
        return node.GetValue()

    @property
    def camera_model(self) -> str:
        return self._cam.TLDevice.DeviceModelName.GetValue()

    @property
    def serial_number(self) -> str:
        return self._cam.TLDevice.DeviceSerialNumber.GetValue()

    @property
    def aspect_ratio(self) -> float:
        w, h = self.resolution
        return w / h

    @property
    def x_mm(self) -> NDArray[np.float64]:
        return np.arange(self.resolution[0]) * self.pixel_pitch_mm

    @property
    def y_mm(self) -> NDArray[np.float64]:
        return np.arange(self.resolution[1]) * self.pixel_pitch_mm

    def get_metadata(self) -> dict[str, Any]:
        """Numeric camera state, ready to pass into provenance.stamp() params."""
        w, h = self.resolution
        return {
            "camera_model": self.camera_model,
            "serial_number": self.serial_number,
            "width_px": w,
            "height_px": h,
            "fov_horizontal_mm": self._fov_horizontal,
            "pixel_pitch_mm": self.pixel_pitch_mm,
            "exposure_us": self.exposure_value,
            "gain_db": self.gain_value,
            "pixel_format": self._cam.PixelFormat.GetCurrentEntry().GetSymbolic(),
            "adc_bit_depth": self._cam.AdcBitDepth.GetCurrentEntry().GetSymbolic(),
            "trigger": self._trigger_wait_ms > 0,
            "temperature_c": self.temperature_c,
            "black_level_percent": self._cam.BlackLevel.GetValue(),
            "black_level_clamping": self._cam.BlackLevelClampingEnable.GetValue(),
        }

    # ------------------------------------------------------------------ #
    # acquisition
    # ------------------------------------------------------------------ #

    def _default_timeout_ms(self) -> int:
        return int(self.exposure_value / 1000) + 1000 + self._trigger_wait_ms

    def _begin(self) -> None:
        if self._cam.IsStreaming():
            raise RuntimeError("Camera is already streaming; something didn't end acquisition")
        self._cam.BeginAcquisition()

    def _end(self) -> None:
        if self._cam is not None and self._cam.IsStreaming():
            self._cam.EndAcquisition()

    def _grab(self, timeout_ms: int | None,
              out: NDArray[np.uint16] | None = None) -> tuple[NDArray[np.uint16], int]:
        """Grab one image, (y, x), and its camera timestamp in ns."""
        if timeout_ms is None:
            timeout_ms = self._default_timeout_ms()
        image = self._cam.GetNextImage(timeout_ms)       # raises SpinnakerException on timeout
        try:
            if image.IsIncomplete():
                raise IncompleteImage(f"Incomplete image: "
                                      f"{describe_image_status(image.GetImageStatus())}")
            arr = image.GetNDArray()                     # (y, x), view into driver buffer
            if out is None:
                out = np.empty(arr.shape, dtype=arr.dtype)
            np.copyto(out, arr)        # copy BEFORE Release: the buffer goes back to the driver
            ts_ns = image.GetChunkData().GetTimestamp()
        finally:
            image.Release()
        return out, ts_ns

    def get_frame(self, timeout_ms: int | None = None) -> NDArray[np.uint16]:
        """Acquire one frame, returned as a contiguous uint16 array indexed (y, x)."""
        self._begin()
        try:
            frame, _ = self._grab(timeout_ms)
        finally:
            self._end()
        return frame

    def get_frame_sequence(self, num_frames: int,
                           timeout_ms: int | None = None,
                           progress=None,
                           max_dropped: int = 0) -> dict:
        """Acquire num_frames frames in one stream: arm() and collect() in one call.

        Use it when nothing has to happen between starting the stream and
        reading it. With the trigger on, it waits for the pulse like collect().
        Arguments and the returned dict are described in collect().
        """
        # Check first, so a bad argument never starts the stream.
        if num_frames < 1:
            raise ValueError("num_frames must be >= 1")
        self.arm()
        return self.collect(num_frames, timeout_ms=timeout_ms,
                            progress=progress, max_dropped=max_dropped)

    def arm(self) -> None:
        """Start the stream and return immediately.

        With the trigger on, the camera now waits for the pulse on Line0;
        with the trigger off it starts taking frames at once. Frames wait in
        the host buffers (STREAM_BUFFERS of them) until collect() reads them.
        """
        # Read the temperature before streaming starts: no effect on timing.
        self._temperature_before = self.temperature_c
        self._begin()

    def collect(self, num_frames: int,
                timeout_ms: int | None = None,
                progress=None,
                max_dropped: int = 0) -> dict:
        """Read num_frames frames from the stream started by arm(), then stop the stream.

        timeout_ms  : per-frame wait; the default includes the trigger wait,
                      so the first frame can wait for the pulse.
        progress    : optional function called after every frame as
                      progress(frames_done, num_frames), e.g. to draw a progress bar.
                      Keep it quick: it runs between frames.
        max_dropped : how many damaged (incomplete) frames may be thrown away and
                      replaced by the next frame in the stream. The default 0
                      raises at the first one. Allow a few for stacks of
                      equivalent frames (flats, darks). Keep 0 when every frame
                      matters, e.g. a time series or a synchronized sweep.

        Returns dict with
            images  : uint16 (frame, y, x)
            time_us : float64, camera timestamps relative to the first frame
                      (a replaced or lost frame shows up as a longer gap)
            dropped : number of damaged frames that were replaced
            x_mm, y_mm : pixel coordinates
            temperature_before_c : camera temperature before acquisition
            temperature_after_c : camera temperature after acquisition
        """
        if not self._cam.IsStreaming():
            raise RuntimeError("collect() needs a running stream: call arm() first")
        if num_frames < 1:
            self._end()          # do not leave the camera streaming
            raise ValueError("num_frames must be >= 1")

        width, height = self.resolution
        warn_if_large(num_frames, width, height)
        images = np.empty((num_frames, height, width), dtype=np.uint16)
        t_ns = np.empty(num_frames, dtype=np.int64)

        collected = 0
        dropped = 0
        try:
            while collected < num_frames:
                try:
                    # Frames are written straight into the array: no extra copy.
                    _, t_ns[collected] = self._grab(timeout_ms, out=images[collected])
                except IncompleteImage:
                    # Damaged frame: count it, and grab the next one in its place.
                    dropped += 1
                    if dropped > max_dropped:
                        raise
                    continue

                collected += 1
                if progress is not None:
                    progress(collected, num_frames)
        finally:
            self._end()      # stops the stream; frames still in flight are discarded
        temperature_after = self.temperature_c

        time_us = (t_ns - t_ns[0]) / 1e3
        warn_if_gaps(time_us)

        return {
            "images": images,
            "time_us": time_us,
            "dropped": dropped,
            "x_mm": self.x_mm,
            "y_mm": self.y_mm,
            "temperature_before_c": self._temperature_before,
            "temperature_after_c": temperature_after,
        }

    # ------------------------------------------------------------------ #
    # teardown
    # ------------------------------------------------------------------ #

    def _close_camera(self) -> None:
        if self._cam is None:
            return
        try:
            self._end()
            if self._cam.IsInitialized():
                self._cam.DeInit()
        finally:
            self._cam = None   # every camera reference must be gone before ReleaseInstance

    def _release(self) -> None:
        try:
            self._close_camera()
        finally:
            if self._cam_list is not None:
                self._cam_list.Clear()
                self._cam_list = None
            if self._system is not None:
                self._system.ReleaseInstance()
                self._system = None

    def deinit(self) -> None:
        """Release the camera and the Spinnaker system. Safe to call twice."""
        self._release()


if __name__ == "__main__":
    # Bench self-test:  python -m labtools.flir
    with FLIRCamera(exposure_value=10_000, gain_value=0) as cam:
        print(cam.get_metadata())
        f = cam.get_frame()
        print("frame", f.shape, f.dtype, "min/max", f.min(), f.max())
        s = cam.get_frame_sequence(10)
        dt = np.diff(s["time_us"])
        print("sequence", s["images"].shape, f"frame interval {dt.mean():.1f} ± {dt.std():.1f} us")