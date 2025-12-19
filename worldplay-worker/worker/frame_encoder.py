"""
Frame Encoder - encodes video frames for streaming.

Handles H.264 encoding of numpy frames for efficient WebSocket transmission.
"""

import asyncio
import io
from typing import Optional
import numpy as np


class FrameEncoder:
    """
    Encodes video frames for network transmission.

    Supports:
    - H.264 encoding (via ffmpeg/PyAV)
    - JPEG fallback for simpler deployment
    - Raw binary for lowest latency
    """

    def __init__(
        self,
        codec: str = "jpeg",  # "h264", "jpeg", or "raw"
        quality: int = 85,
        use_hardware: bool = True
    ):
        """
        Initialize the frame encoder.

        Args:
            codec: Encoding codec ("h264", "jpeg", or "raw")
            quality: Quality for lossy compression (1-100)
            use_hardware: Use hardware encoding if available
        """
        self.codec = codec
        self.quality = quality
        self.use_hardware = use_hardware

        # Check for encoding libraries
        self._check_dependencies()

        # H.264 encoder state
        self.h264_encoder = None

    def _check_dependencies(self):
        """Check available encoding libraries."""
        self.has_av = False
        self.has_cv2 = False
        self.has_pillow = False

        try:
            import av
            self.has_av = True
        except ImportError:
            pass

        try:
            import cv2
            self.has_cv2 = True
        except ImportError:
            pass

        try:
            from PIL import Image
            self.has_pillow = True
        except ImportError:
            pass

        # Determine best available codec
        if self.codec == "h264" and not self.has_av:
            print("PyAV not available, falling back to JPEG")
            self.codec = "jpeg"

        if self.codec == "jpeg" and not (self.has_cv2 or self.has_pillow):
            print("No image library available, using raw encoding")
            self.codec = "raw"

    async def encode(self, frame: np.ndarray) -> bytes:
        """
        Encode a video frame for transmission.

        Args:
            frame: Numpy array (H, W, 3) in RGB format

        Returns:
            Encoded frame data as bytes
        """
        # Run encoding in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._encode_sync, frame)

    def _encode_sync(self, frame: np.ndarray) -> bytes:
        """Synchronous frame encoding."""
        if self.codec == "h264":
            return self._encode_h264(frame)
        elif self.codec == "jpeg":
            return self._encode_jpeg(frame)
        else:
            return self._encode_raw(frame)

    def _encode_h264(self, frame: np.ndarray) -> bytes:
        """Encode frame as H.264."""
        if not self.has_av:
            return self._encode_jpeg(frame)

        import av

        # Initialize encoder if needed
        if self.h264_encoder is None:
            self.h264_encoder = H264Encoder(
                width=frame.shape[1],
                height=frame.shape[0],
                fps=24,
                use_hardware=self.use_hardware
            )

        return self.h264_encoder.encode_frame(frame)

    def _encode_jpeg(self, frame: np.ndarray) -> bytes:
        """Encode frame as JPEG."""
        if self.has_cv2:
            import cv2
            # Convert RGB to BGR for OpenCV
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            _, encoded = cv2.imencode(
                '.jpg',
                bgr_frame,
                [cv2.IMWRITE_JPEG_QUALITY, self.quality]
            )
            return encoded.tobytes()

        elif self.has_pillow:
            from PIL import Image
            img = Image.fromarray(frame)
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=self.quality)
            return buffer.getvalue()

        else:
            return self._encode_raw(frame)

    def _encode_raw(self, frame: np.ndarray) -> bytes:
        """Encode frame as raw bytes with header."""
        # Simple format: [height:4][width:4][channels:4][data]
        h, w, c = frame.shape
        header = h.to_bytes(4, 'big') + w.to_bytes(4, 'big') + c.to_bytes(4, 'big')
        return header + frame.tobytes()

    async def decode(self, data: bytes) -> np.ndarray:
        """
        Decode frame data back to numpy array.

        Useful for testing and verification.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._decode_sync, data)

    def _decode_sync(self, data: bytes) -> np.ndarray:
        """Synchronous frame decoding."""
        if self.codec == "jpeg":
            return self._decode_jpeg(data)
        elif self.codec == "raw":
            return self._decode_raw(data)
        else:
            # H.264 decoding is more complex
            return self._decode_jpeg(data)

    def _decode_jpeg(self, data: bytes) -> np.ndarray:
        """Decode JPEG frame."""
        if self.has_cv2:
            import cv2
            arr = np.frombuffer(data, dtype=np.uint8)
            bgr_frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

        elif self.has_pillow:
            from PIL import Image
            img = Image.open(io.BytesIO(data))
            return np.array(img)

        else:
            raise RuntimeError("No image library available for decoding")

    def _decode_raw(self, data: bytes) -> np.ndarray:
        """Decode raw frame."""
        h = int.from_bytes(data[:4], 'big')
        w = int.from_bytes(data[4:8], 'big')
        c = int.from_bytes(data[8:12], 'big')
        frame_data = data[12:]
        return np.frombuffer(frame_data, dtype=np.uint8).reshape((h, w, c))


class H264Encoder:
    """
    H.264 encoder using PyAV.

    Provides efficient video encoding for streaming.
    """

    def __init__(
        self,
        width: int,
        height: int,
        fps: int = 24,
        bitrate: int = 2_000_000,
        use_hardware: bool = True
    ):
        """
        Initialize H.264 encoder.

        Args:
            width: Frame width
            height: Frame height
            fps: Frames per second
            bitrate: Target bitrate in bits/second
            use_hardware: Use hardware encoding if available
        """
        import av

        self.width = width
        self.height = height
        self.fps = fps

        # Create output container (in-memory)
        self.output = io.BytesIO()
        self.container = av.open(self.output, mode='w', format='h264')

        # Select encoder
        codec_name = 'h264'
        if use_hardware:
            # Try hardware encoders
            for hw_codec in ['h264_nvenc', 'h264_videotoolbox', 'h264_vaapi']:
                try:
                    av.codec.Codec(hw_codec, 'w')
                    codec_name = hw_codec
                    print(f"Using hardware encoder: {hw_codec}")
                    break
                except av.codec.UnknownCodecError:
                    continue

        # Create stream
        self.stream = self.container.add_stream(codec_name, rate=fps)
        self.stream.width = width
        self.stream.height = height
        self.stream.pix_fmt = 'yuv420p'
        self.stream.bit_rate = bitrate

        # Encoding options for low latency
        self.stream.options = {
            'tune': 'zerolatency',
            'preset': 'ultrafast'
        }

        self.frame_count = 0

    def encode_frame(self, frame: np.ndarray) -> bytes:
        """
        Encode a single frame.

        Args:
            frame: Numpy array (H, W, 3) in RGB format

        Returns:
            H.264 encoded data
        """
        import av

        # Create VideoFrame
        video_frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
        video_frame = video_frame.reformat(format='yuv420p')
        video_frame.pts = self.frame_count
        video_frame.time_base = av.Fraction(1, self.fps)

        # Encode
        self.output.seek(0)
        self.output.truncate()

        for packet in self.stream.encode(video_frame):
            self.container.mux(packet)

        self.frame_count += 1

        # Get encoded data
        data = self.output.getvalue()
        return data

    def close(self):
        """Close the encoder and flush remaining data."""
        # Flush encoder
        for packet in self.stream.encode():
            self.container.mux(packet)

        self.container.close()
