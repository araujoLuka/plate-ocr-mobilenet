import os
from time import sleep

def progress_bar(progress: int, total: int, print_it: bool = False) -> str:
    progress_len: int = len(str(total))
    total_len: int = len(str(total))

    percent: float = (progress / total) * 100
    percent_str: str = f"{percent:.2f}%".rjust(7)
    summary: str = f"({progress}/{total})".rjust(1 + progress_len + 3 + total_len)
    terminal_size: int = os.get_terminal_size().columns

    start: str = "\r[ ]"
    if progress == total:
        start = "\r[X]"
    start += " Progress: ["

    end: str = "]"
    end += summary
    end += " "
    end += percent_str

    bar_max: int = terminal_size - len(start) - len(summary) - len(end)
    bar_fill: int = int(bar_max * (progress / total))
    bar_empty: int = bar_max - bar_fill
    bar: str = "=" * bar_fill + " " * bar_empty

    p_bar: str = start + bar + end

    if print_it:
        print(p_bar, end="", flush=True)
        if progress == total:
            print()

    return p_bar

if __name__ == "__main__":
    for i in range(1000):
        progress_bar(i+1, 1000, print_it=True)
        sleep(0.001)
