"""
FLIR (Spinnaker / PySpin) camera driver.

Author: Mykhailo Vorobiov

Conventions
-----------
- Frames are uint16 arrays indexed (y, x), i.e. (row, column), the camera's native
  order; sequences are (frame, y, x). imshow(frame) displays correctly, and
  frame[iy, ix] pairs with (y_mm[iy], x_mm[ix]).
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
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

import PySpin


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
                raise RuntimeError(f"Incomplete image, status {image.GetImageStatus()}")
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
                           progress=None) -> dict[str, NDArray]:
        """Acquire num_frames frames in one stream.

        progress : optional function called after every frame as
                   progress(frames_done, num_frames), e.g. to draw a progress bar.
                   Keep it quick: it runs between frames.

        Returns dict with
            images  : uint16 (frame, y, x)
            time_us : float64, camera timestamps relative to the first frame
            x_mm, y_mm : pixel coordinates
        """
        if num_frames < 1:
            raise ValueError("num_frames must be >= 1")
        w, h = self.resolution
        images = np.empty((num_frames, h, w), dtype=np.uint16)
        t_ns = np.empty(num_frames, dtype=np.int64)

        self._begin()
        try:
            for i in range(num_frames):
                _, t_ns[i] = self._grab(timeout_ms, out=images[i])
                if progress is not None:
                    progress(i + 1, num_frames)
        finally:
            self._end()

        return {
            "images": images,
            "time_us": (t_ns - t_ns[0]) / 1e3,
            "x_mm": self.x_mm,
            "y_mm": self.y_mm,
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
