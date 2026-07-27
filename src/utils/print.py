import json

def print_message(msg: str, show_borders: bool = True, skip_line: bool = True):
    msg_len = len(msg)
    if show_borders:
        print("="*msg_len)
    print(msg)
    if show_borders:
        print("="*msg_len)
    if skip_line:
        print("\n")
        
def pprint(data: dict, indent: int = 4):
    print(json.dumps(data, indent=indent, sort_keys=True))