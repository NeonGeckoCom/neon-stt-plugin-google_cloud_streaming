# NEON AI (TM) SOFTWARE, Software Development Kit & Application Framework
# All trademark and other rights reserved by their respective owners
# Copyright 2008-2022 Neongecko.com Inc.
# Contributors: Daniel McKnight, Guy Daniels, Elon Gasper, Richard Leeds,
# Regina Bloomstine, Casimiro Ferreira, Andrii Pernatii, Kirill Hrymailo
# BSD-3 License
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from this
#    software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS  BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS;  OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE,  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import sys
import unittest

from threading import Event
from unittest import skip

from neon_utils.file_utils import get_audio_file_stream

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from neon_stt_plugin_google_cloud_streaming import GoogleCloudStreamingSTT

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
TEST_PATH = os.path.join(ROOT_DIR, "test_audio")


class NeonSTT(GoogleCloudStreamingSTT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def stream_stop(self):
        if self.stream is not None:
            self.queue.put(None)
            text = self.stream.finalize()
            to_return = [text]
            self.stream.join()
            if hasattr(self.stream, 'transcriptions'):
                to_return = self.stream.transcriptions
            self.stream = None
            self.queue = None
            self.results_event.set()
            return to_return
        return None


class TestGetSTT(unittest.TestCase):
    def test_get_stt_simple(self):
        stt = GoogleCloudStreamingSTT()
        for file in os.listdir(TEST_PATH):
            transcription = os.path.splitext(os.path.basename(file))[0].lower()
            stream = get_audio_file_stream(os.path.join(TEST_PATH, file))

            def read_file():
                stream.file.setpos(0)
                stt.stream_start()
                try:
                    while True:
                        chunk = stream.read(1024)
                        stt.stream_data(chunk)
                except EOFError:
                    pass

            # Check legacy single transcript support
            read_file()
            result = stt.execute(None)
            result = result.lower()
            self.assertIsInstance(result, str, f"Error processing: {file}")
            self.assertEqual(transcription, result)

            # Check alternative transcripts
            read_file()
            transcripts = stt.transcribe()
            self.assertEqual(transcripts[0][0].lower(), result,
                             f"Error processing: {file}")
            for transcript in transcripts:
                self.assertIsInstance(transcript[0], str)
                self.assertIsInstance(transcript[1], float)

    @skip("Neon-specific STT is deprecated")
    def test_get_stt_neon(self):
        results_event = Event()
        stt = NeonSTT(results_event=results_event)
        for file in os.listdir(TEST_PATH):
            transcription = os.path.splitext(os.path.basename(file))[0].lower()
            stream = get_audio_file_stream(os.path.join(TEST_PATH, file))
            stt.stream_start()
            try:
                while True:
                    chunk = stream.read(1024)
                    stt.stream_data(chunk)
            except EOFError:
                pass

            result = stt.execute(None)
            self.assertIsNotNone(result, f"Error processing: {file}")
            self.assertIsInstance(result, list)
            self.assertIn(transcription, (r.lower() for r in result))

    def test_available_languages(self):
        stt = GoogleCloudStreamingSTT()
        self.assertIsInstance(stt.available_languages, set)
        self.assertIn('en-US', stt.available_languages)


if __name__ == '__main__':
    unittest.main()
