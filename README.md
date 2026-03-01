# ChatSummaryBot

ChatSummaryBot is a simple bot that summarizes chat messages in group chats. Currently, it supports Telegram, but it can be easily extended to other platforms.

## Features

- Summarizes chat messages from Telegram channels.
- Supports multiple summarization models: OpenAI, Ollama, and Groq.
- Configurable via a YAML file.
- Generates a daily summary in Ukrainian in a fixed format:
  - `#ПідсумокДня`
  - `🏠 Основні теми`
  - `📝 Що відбувалось`
  - `📊 Статистика`
- Provides message/participant stats and top active contributors.
- Extracts and summarizes links and attachments.
- Bots generate messages and send them every 24 hours.
- Supports channels with subtopics (subforms), processing each subtopic separately and generating summaries per topic.
- Ability to publish summaries to a dedicated Telegram channel:
  - Summary from each search channel is posted as a new topic in the target channel.
  - Requires `summary_channel_id` key in the configuration file.


## Requirements

- Python 3.12 or higher. Other version of Python 3 may work, but not tested.
- Telegram API credentials (API ID and API Hash)
- API key for your selected model provider (`OPENAI_API_KEY` or `GROQ_API_KEY`) or local Ollama setup

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/ChatSummaryBot.git
    cd ChatSummaryBot
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv # or python3.12 -m venv venv 
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Configuration

1. Copy the example configuration file and update it with your details:
    ```sh
    cp config.example.yaml config.yaml
    ```

2. Edit `config.yaml` and fill in the required fields:
    ```yaml
    session_name: chatSummary
    api_id: "YOUR_API_ID"
    api_hash: "YOUR_API_HASH"
    model_provider: openai  # openai | groq | ollama
    model_name: gpt-4o-mini
    max_length: 4000
    summarization_frequency: 24  # in hours
    output_dir: summaries
    channels:
      - id: -1001235860760
        name: IT_TALKS_BENELUX
        language: russian
        context: "This is a channel for software engineers working in the Benelux region, discussing various topics. The channel contains a lot of non-serious discussion, but it's known to be useful for finding a job or getting real insights."
        filters:
          keywords: ["tech", "benelux"]

      - id: -1001157394889
        name: LEETCODE_HEROES
        filters:
          keywords: ["leetcode", "interview"]
    ```

## Model Configuration

### OpenAI (recommended for strict output format)
1. Create an API key at https://platform.openai.com/api-keys
2. Set the environment variable (or place it in a local `.env` file):
    ```sh
    export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
    ```

### Groq Cloud
This option is useful if you prefer Groq-hosted models.

1. Sign up for a Groq Cloud account and obtain your API key.
2. Set the `GROQ_API_KEY` environment variable:
    ```sh
    export GROQ_API_KEY="YOUR_GROQ_API_KEY"
    ```

### Ollama Locally



1. Install the Ollama model locally by following the instructions on the Ollama website.
2. Ensure the model is running and accessible on your local machine.

```sh
brew install ollama
ollama pull llama3.2
ollama serve
```

## Usage

1. Run the summarizer bot after setting the provider key (or without key for local Ollama).
    ```sh
    OPENAI_API_KEY=YOUR_OPENAI_API_KEY python summarizeChat.py
    ```

2. The bot will start processing the configured channels and generate summaries based on the specified frequency.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
