''''
Class for FLIR camera data acquisition

Author: Mykhailo Vorobiov
'''
#%%
import numpy as np
import os
from time import perf_counter
from datetime import datetime
from typing import Literal, Dict, Any, Tuple, Optional
from numpy.typing import NDArray

import PySpin 

from pinqued_tools.spectroscopy.spectrum import SpectralData, Axes2D

class FLIRCamera():
    '''
    Class for FLIR camera data acquisition and control.
    '''
    def __init__(self, 
                 cam_index: int = 0,
                 fov_horizontal = 8.1, # mm
                 exposure_value: float = 40000.0,
                 gain_value: float = 20.0,
                 camera_model = 'Blackfly BFS-PGE-04S2M'
                 ):
        
        self._cam_index = cam_index
        self._fov_horizontal = fov_horizontal
        self._cam_model = camera_model
        self._exposure_value = exposure_value
        self._gain_value = gain_value


        # 1. Get singleton reference to system object
        self._system = PySpin.System.GetInstance()

        # 2. Get camera list
        self._cam_list = self._system.GetCameras()
        num_cameras = self._cam_list.GetSize()

        if num_cameras == 0:
            self._cam_list.Clear()
            self._system.ReleaseInstance()
            raise AssertionError('No cameras detected!')
        
        # 3. Get camera and initialize
        self._cam = self._cam_list.GetByIndex(cam_index)
        self._cam.Init()

        try:
            self._cam.Init()

            # Assuming cam is an initialized PySpin Camera object
            node_map = self._cam.GetNodeMap()

            # 1. Turn off Auto Exposure
            # exposure_auto = PySpin.CEnumerationPtr(node_map.GetNode('ExposureAuto'))
            # exposure_auto.SetIntValue(exposure_auto.GetEntryByName('Off').GetValue())
            self._cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)

            # Set manual exposure time (in microseconds)
            exposure_time = PySpin.CFloatPtr(node_map.GetNode('ExposureTime'))
            exposure_time.SetValue(self._exposure_value) 

            # 2. Turn off Auto Gain
            gain_auto = PySpin.CEnumerationPtr(node_map.GetNode('GainAuto'))
            gain_auto.SetIntValue(gain_auto.GetEntryByName('Off').GetValue())

            # Set manual gain (in dB)
            gain = PySpin.CFloatPtr(node_map.GetNode('Gain'))
            gain.SetValue(self._gain_value) # Example: 0 dB

            # 3. Disable Gamma
            gamma_enable = PySpin.CBooleanPtr(node_map.GetNode('GammaEnable'))
            gamma_enable.SetValue(False) # Set to False to disable, True to enable

            # 1. Enable Chunk Mode
            self._cam.ChunkModeActive.SetValue(True)

            # 2. Enable Timestamp Chunk
            self._cam.ChunkSelector.SetValue(PySpin.ChunkSelector_Timestamp)
            self._cam.ChunkEnable.SetValue(True)

            print('Camera initialized')
        except PySpin.SpinnakerException as ex:
            print(f'Error: {ex}')

    @property
    def index(self) -> int:
        return self._cam_index
    
    @property
    def resolution(self) -> Tuple[int, int]:
        width = self._cam.Width.GetValue()
        height = self._cam.Height.GetValue()
        return width, height
    
    @property
    def fov_horizontal(self) -> float:
        return self._fov_horizontal
    
    @property
    def exposure_value(self) -> float:
        return self._exposure_value
    
    @property
    def gain_value(self) -> float:
        return self._gain_value
    
    @property
    def camera_model(self) -> str:
        return self._cam_model
    
    @property
    def aspect_ratio(self) -> float:
        '''
        Returns current aspect ratio of the camera.
        This will reflect any changes in ROI size set up in SpinView.
        '''
        return self.resolution[0] / self.resolution[1]
    
    def set_px_format_mono16bit(self):
        self._cam.AdcBitDepth.SetValue(PySpin.AdcBitDepth_Bit12)
        self._cam.PixelFormat.SetValue(PySpin.PixelFormat_Mono16)

    def switch(self, cam_index: int):
        if self._cam is not None:
            self._cam.DeInit()
            del self._cam
        if self._cam_index != cam_index:
            self._cam_index = cam_index
            self._cam = self._cam_list.GetByIndex(cam_index)

    def deinit(self):
        self._cam.DeInit()
        del self._cam
        self._cam_list.Clear()
        self._system.ReleaseInstance()
        print('Camera deinitialized')
        
    def trigger(self, on_off: Literal['on', 'off'] = 'on'):
        if on_off == 'on':
            self._cam.TriggerSelector.SetValue(PySpin.TriggerSelector_AcquisitionStart)
            self._cam.TriggerSource.SetValue(PySpin.TriggerSource_Line0) # Or Line0-2 depending on hardware
            self._cam.TriggerActivation.SetValue(PySpin.TriggerActivation_RisingEdge)
            self._cam.TriggerMode.SetValue(PySpin.TriggerMode_On)
        elif on_off == 'off':
            if self._cam.IsStreaming():
                self._cam.EndAcquisition()
            self._cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)
            self._cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
    
    def disable_fps_limit(self):
        if hasattr(self._cam, 'AcquisitionFrameRateEnable'):
            self._cam.AcquisitionFrameRateEnable.SetValue(False)
        elif hasattr(self._cam, 'AcquisitionFrameRateEnabled'):
            self._cam.AcquisitionFrameRateEnabled.SetValue(False)

    def end_acqusition(self):
        self._cam.EndAcquisition()
        print('Acquisition ended')

    def acquire_sequence(self, num_frames: int):
        self.set_px_format_mono16bit()

        # 1. Enable chunk data
        nodemap = self._cam.GetNodeMap()
        chunk_mode = PySpin.CBooleanPtr(nodemap.GetNode('ChunkModeActive'))
        if chunk_mode.GetAccessMode() == PySpin.RW:
            chunk_mode.SetValue(True)

        # Enable timestamp
        chunk_selector = PySpin.CEnumerationPtr(nodemap.GetNode('ChunkSelector'))
        chunk_entry = chunk_selector.GetEntryByName('Timestamp')
        chunk_selector.SetIntValue(chunk_entry.GetValue())
        chunk_enable = PySpin.CBooleanPtr(nodemap.GetNode('ChunkEnable'))
        chunk_enable.SetValue(True)
        print('Timestamp chunk enabled')

        # 2. Acquire images and get timestamp
        self._cam.BeginAcquisition()
        print(f'Camera acquisition started. Acquiring {num_frames} frames...')

        time = np.zeros(num_frames)
        images = np.empty((num_frames, self.resolution[1], self.resolution[0]), dtype=np.uint16)
        for i in range(num_frames):
            image = self._cam.GetNextImage()
            images[i] = image.GetNDArray()


            # Get the chunk data container
            chunk_data = image.GetChunkData()
            
            # Get timestamp in nanoseconds
            timestamp = chunk_data.GetTimestamp()

            # 3. Conver timestamps to milliseconds
            print(f"Time stamp: {timestamp:.4f} ms")

            time[i] = timestamp
            image.Release()

        self.end_acqusition()

        # Swap axes 1 and 2 to comply with (f, x, y) convention of the rest of the code.
        # Contiguous array is requored to prevent slow processing down the pipeline.
        images = np.ascontiguousarray(np.swapaxes(images, 1, 2))
        time -= time[0]
        
        # Prepare data for return
        # Assemble metadata dictionary
        metadata = {'Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'Camera': self._cam_model,
                    'Resolution': f'{self.resolution[0]}x{self.resolution[1]}',
                    'FOV (mm)': f'{self._fov_horizontal}',
                    'Exposure time (us)': f'{self._exposure_value}',
                    'Gain (dB)': f'{self._gain_value}'
                    }
        # Create data container to return
        data = SpectralData(signal=images,
                            axes=Axes2D(f=time, 
                                        x=np.linspace(0, self._fov_horizontal, self.resolution[0]),
                                        y=np.linspace(0, self._fov_horizontal/self.aspect_ratio, self.resolution[1]),
                                        units={'f':'us', 'x': 'mm', 'y': 'mm'}),
                            units={'signal': 'counts'},
                            metadata=metadata)
        return data



#%%
# -------------- Usage example -------------
if __name__=='__main__':
    # Apply custom plotting style
    from matplotlib import pyplot as plt
    from pinqued_tools.spectroscopy.spectrum import SpectralDataProcessor
    from pinqued_tools.analysis.plotting import set_mpl_style
    set_mpl_style()

    sweep_hz = 0.02
    exposure_ms = 40 *1000 # us 
    num_frames = np.floor(0.5 / (sweep_hz * exposure_ms*1e-6)).astype(int)
    print(f'Number of frames: {num_frames}')

    cam = FLIRCamera(cam_index=0,
                     exposure_value=exposure_ms)
    cam.trigger('off')
    cam.disable_fps_limit()
    cam_data = cam.acquire_sequence(num_frames)
    cam.deinit()

    print(cam_data)

    plt.figure(1)
    plt.pcolormesh(cam_data.axes.x, cam_data.axes.f, cam_data.signal[:,:,100], cmap='jet')

    plt.figure(2)
    plt.pcolormesh(cam_data.axes.x, cam_data.axes.y, cam_data.signal[1].T, cmap='jet')

    sproc = SpectralDataProcessor(cam_data)
    sproc.preprocess() # convert to relative dip intensity and calculate resulting error
    sproc.remove_fmean()
    sproc.bin(px_per_bin=18, axis=1)
    sproc.data
    plt.figure(3)
    plt.pcolormesh(sproc.data.axes.x, sproc.data.axes.f, sproc.data.signal[:,:,50], cmap='jet')

    print(sproc.data)
    
    from pinqued_tools.data.io import SpectralDataH5Handler
    h5_handler = SpectralDataH5Handler()
    h5_handler.save(data=sproc.data, 
                    file_path='processed_file.h5')
# %%
