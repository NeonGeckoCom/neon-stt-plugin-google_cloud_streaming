# NeonAI Google Streaming STT Plugin
[Mycroft](https://mycroft-ai.gitbook.io/docs/mycroft-technologies/mycroft-core/plugins) compatible
STT Plugin for Google Speech-to-Text.

# Configuration:
A json credential can be ottained following [instructions here](https://developers.google.com/workspace/guides/create-credentials#create_credentials_for_a_service_account) should be saved at: `~/.local/share/neon/google.json`.
JSON Credentials may alternatively be included in the tts configuration as shown below.

```yaml
tts:
    module: google_cloud_streaming
    google_cloud: {lang: en-us, credential: None}
```
