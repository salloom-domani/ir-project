import re


def sanitize(text: str) -> str:
    # Replace disallowed characters with underscore
    text = re.sub(r"[^a-zA-Z0-9._-]", "_", text)

    # Strip non-alphanumeric characters from start and end
    text = re.sub(r"^[^a-zA-Z0-9]+", "", text)
    text = re.sub(r"[^a-zA-Z0-9]+$", "", text)

    # Ensure minimum and maximum length
    if len(text) < 3:
        text = text.ljust(3, "x")  # pad with 'x'
    elif len(text) > 512:
        text = text[:512]

    return text


if __name__ == "__main__":
    result = sanitize("antique/test")
    print(result)
