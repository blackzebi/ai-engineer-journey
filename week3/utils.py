import json, logging, re

logging.basicConfig(level=logging.INFO)

def parse_json_safely(raw: str, default=None):
    if not raw:
        return default

    text = raw.strip()

    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logging.warning("parse_json_safely loi: %s | raw=%r", e, raw[:120])
        return default