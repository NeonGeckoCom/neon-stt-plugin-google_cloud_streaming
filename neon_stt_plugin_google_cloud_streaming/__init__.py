# NEON AI (TM) SOFTWARE, Software Development Kit & Application Development System
# All trademark and other rights reserved by their respective owners
# Copyright 2008-2021 Neongecko.com Inc.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions
#    and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
#    and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
#    products derived from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from queue import Queue
from threading import Event

from google.cloud import speech
from google.oauth2.service_account import Credentials
from ovos_utils.log import LOG
from ovos_plugin_manager.templates.stt import StreamingSTT, StreamThread


class GoogleCloudStreamingSTT(StreamingSTT):
    """
        Streaming STT interface for Google Cloud Speech-To-Text
        To use pip install google-cloud-speech and add the
        Google API key to local mycroft.conf file. The STT config
        will look like this:

        "stt": {
            "module": "google_cloud_streaming",
            "google_cloud_streaming": {
                "credential": {
                    "json": {
                        # Paste Google API JSON here
        ...

    """

    def __init__(self, config=None, **kwargs):
        super(GoogleCloudStreamingSTT, self).__init__(config=config)
        self.results_event = kwargs.get("results_event")

        # override language with module specific language selection
        self.language = self.config.get('lang') or self.lang
        self.queue = None

        if self.credential:
            if not self.credential.get("json"):
                self.credential["json"] = self.credential
            LOG.debug(f"Got credentials: {self.credential}")
            credentials = Credentials.from_service_account_info(
                self.credential.get('json')
            )
        else:
            try:
                from neon_utils.authentication_utils import find_neon_google_keys
                credential_json = find_neon_google_keys()
                credentials = Credentials.from_service_account_info(credential_json)
            except Exception as e:
                LOG.error(e)
                credentials = None
        self.client = speech.SpeechClient(credentials=credentials)
        recognition_config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code=self.language,
            max_alternatives=3
        )
        self.streaming_config = speech.StreamingRecognitionConfig(
            config=recognition_config,
            interim_results=False
        )

    def create_streaming_thread(self):
        self.queue = Queue()
        return GoogleStreamThread(
            self.queue,
            self.language,
            self.client,
            self.streaming_config,
            self.results_event
        )


class GoogleStreamThread(StreamThread):
    def __init__(self, queue, lang, client, streaming_config, results_event=None):
        super().__init__(queue, lang)
        self.name = "StreamThread"
        self.client = client
        self.streaming_config = streaming_config
        self.results_event = results_event or Event()
        self.transcriptions = []

    def handle_audio_stream(self, audio, language):
        req = (speech.StreamingRecognizeRequest(audio_content=x) for x in audio)
        responses = self.client.streaming_recognize(self.streaming_config, req)
        # Responses are yielded, but we will return once the first sentence is transcribed
        for res in responses:
            for result in res.results:
                LOG.debug(result)
            if res.results and res.results[0].is_final:
                self.transcriptions = []
                for alternative in res.results[0].alternatives:
                    transcription = alternative.transcript
                    self.transcriptions.append(transcription)
        LOG.debug(self.transcriptions)
        self.results_event.set()
        if self.transcriptions:
            self.text = self.transcriptions[0]  # Mycroft compat.
        return self.transcriptions

    def finalize(self):
        self.results_event.wait()
        return super().finalize()
