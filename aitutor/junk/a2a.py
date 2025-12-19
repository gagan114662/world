"""
## Documentation
Quickstart: https://github.com/google-gemini/cookbook/blob/main/quickstarts/Get_started_LiveAPI.py

## Setup

To install the dependencies for this script, run:

```
pip install google-genai pyaudio
```

## API Key Setup
Set environment variable before running:
Windows PowerShell: $env:GOOGLE_API_KEY="your-api-key"
Windows CMD: set GOOGLE_API_KEY=your-api-key
Linux/Mac: export GOOGLE_API_KEY="your-api-key"

Python 3.10.6 compatible version
"""

import os
import asyncio
import traceback

import pyaudio

from google import genai
from google.genai import types

FORMAT = pyaudio.paInt16
CHANNELS = 1
RECEIVE_SAMPLE_RATE = 24000

MODEL = "models/gemini-2.5-flash-native-audio-preview-09-2025"

# Try multiple environment variable names for API key
api_key = "AIzaSyD9rkM_Z3UK92634LwxI4l6T4D-qRn4jyg"

if not api_key:
    raise ValueError(
        "Google API key not found! Please set one of these environment variables:\n"
        "  Windows PowerShell: $env:GOOGLE_API_KEY=\"your-api-key\"\n"
        "  Windows CMD: set GOOGLE_API_KEY=your-api-key\n"
        "  Linux/Mac: export GOOGLE_API_KEY=\"your-api-key\"\n"
    )

client = genai.Client(
    http_options={"api_version": "v1beta"},
    api_key=api_key,
)

tools = [
    types.Tool(
        function_declarations=[
        ]
    ),
]

CONFIG = types.LiveConnectConfig(
    response_modalities=[
        "AUDIO",
    ],
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Schedar")
        )
    ),
    realtime_input_config=types.RealtimeInputConfig(
        automatic_activity_detection=types.AutomaticActivityDetection(
            disabled=True  # Disable automatic detection to use manual activity signals
        )
    ),
    context_window_compression=types.ContextWindowCompressionConfig(
        trigger_tokens=25600,
        sliding_window=types.SlidingWindow(target_tokens=12800),
    ),
    tools=tools,
    system_instruction=types.Content(
        parts=[types.Part.from_text(text="speak with user in gujrati if user not silent for 5 second then ask from your sode are yopu still their?")],
        role="user"
    ),
)

pya = pyaudio.PyAudio()


class AudioLoop:
    def __init__(self):
        self.audio_in_queue = None
        self.session = None
        self.should_stop = False

    async def send_text(self,interupt=True):
        """Handle text input from user with interruption support"""
        while not self.should_stop:
            try:
                text = await asyncio.to_thread(
                    input,
                    "message > ",
                )

                if text.lower() == "q":
                    self.should_stop = True
                    break
                
                if interupt:
                    # Clear audio queue to stop current playback (interruption)
                    while not self.audio_in_queue.empty():
                        self.audio_in_queue.get_nowait()
                
                # Send user input using activity signals (not deprecated .send())
                await self.session.send_realtime_input(activity_start=types.ActivityStart())
                await self.session.send_realtime_input(text=text or ".")
                await self.session.send_realtime_input(activity_end=types.ActivityEnd())
            except Exception as e:
                print(f"Error in send_text: {e}")
                break

    async def receive_audio(self):
        """Receive audio responses from the model"""
        while not self.should_stop:
            try:
                turn = self.session.receive()
                async for response in turn:
                    if data := response.data:
                        self.audio_in_queue.put_nowait(data)
                        continue
                    if text := response.text:
                        print(text, end="")

                # If you interrupt the model, it sends a turn_complete.
                # For interruptions to work, we need to stop playback.
                # So empty out the audio queue because it may have loaded
                # much more audio than has played yet.
                while not self.audio_in_queue.empty():
                    self.audio_in_queue.get_nowait()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in receive_audio: {e}")
                break

    async def play_audio(self):
        """Play audio responses"""
        stream = None
        try:
            stream = await asyncio.to_thread(
                pya.open,
                format=FORMAT,
                channels=CHANNELS,
                rate=RECEIVE_SAMPLE_RATE,
                output=True,
            )
            while not self.should_stop:
                bytestream = await self.audio_in_queue.get()
                await asyncio.to_thread(stream.write, bytestream)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Error in play_audio: {e}")
        finally:
            if stream:
                stream.close()

    async def run(self):
        """Main run loop"""
        try:
            async with client.aio.live.connect(model=MODEL, config=CONFIG) as session:
                self.session = session
                self.audio_in_queue = asyncio.Queue()

                # Create all tasks
                tasks = [
                    asyncio.create_task(self.send_text()),
                    asyncio.create_task(self.receive_audio()),
                    asyncio.create_task(self.play_audio()),
                ]

                # Wait for send_text to complete (user types 'q')
                await tasks[0]
                
                # Signal other tasks to stop
                self.should_stop = True
                
                # Cancel all other tasks
                for task in tasks[1:]:
                    task.cancel()
                
                # Wait for tasks to complete
                await asyncio.gather(*tasks[1:], return_exceptions=True)

        except asyncio.CancelledError:
            print("\nShutting down...")
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"Error in main loop: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    print("Text-to-Audio Gemini Live API")
    print("Type 'q' to quit")
    print("-" * 40)
    
    main = AudioLoop()
    asyncio.run(main.run())
