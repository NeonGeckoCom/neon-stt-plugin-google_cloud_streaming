#!/usr/bin/env bash

# NEON AI (TM) SOFTWARE, Software Development Kit & Application Development System
#
# Copyright 2008-2020 Neongecko.com Inc. | All Rights Reserved
#
# Notice of License - Duplicating this Notice of License near the start of any file containing
# a derivative of this software is a condition of license for this software.
# Friendly Licensing:
# No charge, open source royalty free use of the Neon AI software source and object is offered for
# educational users, noncommercial enthusiasts, Public Benefit Corporations (and LLCs) and
# Social Purpose Corporations (and LLCs). Developers can contact developers@neon.ai
# For commercial licensing, distribution of derivative works or redistribution please contact licenses@neon.ai
# Distributed on an "AS IS” basis without warranties or conditions of any kind, either express or implied.
# Trademarks of Neongecko: Neon AI(TM), Neon Assist (TM), Neon Communicator(TM), Klat(TM)
# Authors: Guy Daniels, Daniel McKnight, Regina Bloomstine, Elon Gasper, Richard Leeds
#
# Specialized conversational reconveyance options from Conversation Processing Intelligence Corp.
# US Patents 2008-2020: US7424516, US20140161250, US20140177813, US8638908, US8068604, US8553852, US10530923, US10530924
# China Patent: CN102017585  -  Europe Patent: EU2156652  -  Patents Pending

from queue import Queue
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types
from google.oauth2.service_account import Credentials

from mycroft.stt import StreamingSTT, StreamThread
from mycroft.util.log import LOG


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

    def __init__(self, results_event=None):
        super(GoogleCloudStreamingSTT, self).__init__(results_event)
        # override language with module specific language selection
        self.language = self.config.get('lang') or self.lang
        self.alt_langs = self.conf['speech']['alt_languages']
        self.queue = None

        if not self.credential.get("json"):
            self.credential["json"] = self.credential
        if self.credential.get("json"):
            credentials = Credentials.from_service_account_info(
                self.credential.get('json')
            )
        else:
            import os
            cred_file = os.path.expanduser("~/.local/share/neon/google.json")
            if os.path.isfile(cred_file):
                credentials = Credentials.from_service_account_file(cred_file)
            else:
                credentials = None
        self.client = speech.SpeechClient(credentials=credentials)
        recognition_config = types.RecognitionConfig(
            encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code=self.language,
            alternative_language_codes=self.alt_langs,
            # model='command_and_search',
            max_alternatives=3
        )
        self.streaming_config = types.StreamingRecognitionConfig(
            config=recognition_config,
            interim_results=False
            # single_utterance=True,
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
        self.client = client
        self.streaming_config = streaming_config
        self.results_event = results_event
        self.transcriptions = []

    def handle_audio_stream(self, audio, language):
        req = (types.StreamingRecognizeRequest(audio_content=x) for x in audio)
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
            if self.results_event:
                self.results_event.set()
            return self.transcriptions
