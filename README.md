# Code Agent

Python 기반 coding agent입니다. `uv`로 실행하며 Anthropic, OpenAI, Google Gemini provider를 선택할 수 있습니다.

## Requirements

- Python `>=3.10,<3.14`
- Provider별 API key
  - Anthropic: `ANTHROPIC_API_KEY`
  - OpenAI: `OPENAI_API_KEY`
  - Google Gemini: `GEMINI_API_KEY` 또는 `GOOGLE_API_KEY`

## Setup

```bash
uv sync
```

## Run

기본 provider는 `anthropic`입니다.

```bash
uv run python main.py "현재 디렉터리 구조 설명해줘"
```

특정 provider를 지정하려면:

```bash
uv run python main.py --provider openai --model gpt-4.1 "테스트 코드 추가해줘"
uv run python main.py --provider google --model gemini-2.5-pro "리팩터링 포인트 찾아줘"
```

REPL 모드에서는 슬래시 명령으로 런타임 설정을 바꿀 수 있습니다.

```bash
/provider openai
/model gpt-5.4
/max-turns 20
/config
```

환경 변수로도 설정할 수 있습니다.

```bash
export CODE_AGENT_PROVIDER=openai
export CODE_AGENT_MODEL=gpt-4.1
export CODE_AGENT_SUMMARY_MODEL=gpt-4.1-mini
export CODE_AGENT_SUBAGENT_MODEL=gpt-4.1-mini
```

## Supported Provider Values

- `anthropic`
- `openai`
- `google`
- Alias: `claude -> anthropic`, `gemini -> google`
