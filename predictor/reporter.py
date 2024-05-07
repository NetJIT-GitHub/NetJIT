from __future__ import annotations

import queue
import threading
import typing


class Reporter(typing.Protocol):
    def __call__(self, content: typing.Any) -> None:
        pass

    def __enter__(self) -> Reporter:
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class FileReporter:
    def __init__(self, file: str):
        self.file = file
        self.queue = queue.Queue()

    def __call__(self, content: typing.Any):
        self.queue.put(content)

    def __enter__(self):
        threading.Thread(target=self.run).start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.queue.put(None)
        self.queue.join()

    def run(self):
        with open(self.file, 'w') as f:
            while True:
                data = self.queue.get()
                if data is None:
                    self.queue.task_done()
                    break
                f.write(data)
                self.queue.task_done()
