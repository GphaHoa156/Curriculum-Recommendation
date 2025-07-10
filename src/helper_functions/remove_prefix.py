import re
def remove_prefix(text):
    """
        Remove any unesscessary part before the real content of text
        For example: 2.3 Text -> Text
    """
    # Loại mọi thứ đứng trước và bao gồm chuỗi đánh số dạng 1. / 1.2. / 1.2.3. / v.v.
    pattern = r'^\s*[^a-zA-Z0-9]{0,5}?(?:\w+\s+)*?(\d+(?:\.\d+)*)(\s*[.:])\s*\d*\s*'

    match = re.match(pattern, text)
    if match:
        return text[match.end():].strip()
    return text.strip()