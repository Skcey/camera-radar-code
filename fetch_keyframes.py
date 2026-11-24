"""Resume a public nuScenes archive using verified HTTP byte ranges on Linux."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time
from urllib.request import Request, urlopen


def main(args):
    with urlopen(Request(args.url, method='HEAD'), timeout=20) as response:
        size = int(response.headers['Content-Length'])
    state = args.out.with_suffix(args.out.suffix + '.ranges.json')
    if state.exists():
        record = json.loads(state.read_text())
        if record['size'] != size or record['url'] != args.url:
            raise ValueError('Resume metadata does not match the requested archive')
    else:
        prefix = args.out.stat().st_size if args.out.exists() else 0
        if prefix > size:
            raise ValueError('Existing download is longer than remote archive')
        record = {'url': args.url, 'size': size, 'prefix': prefix, 'done': []}
        state.write_text(json.dumps(record))
    chunk_size = 64 * 1024 * 1024
    starts = list(range(record['prefix'], size, chunk_size))
    done = set(record['done'])
    fd = os.open(args.out, os.O_RDWR | os.O_CREAT, 0o644)
    os.ftruncate(fd, size)

    def block(start):
        end = min(start + chunk_size, size) - 1
        for attempt in range(5):
            try:
                req = Request(args.url, headers={'Range': f'bytes={start}-{end}'})
                with urlopen(req, timeout=25) as response:
                    if response.status != 206 or response.headers['Content-Range'] != f'bytes {start}-{end}/{size}':
                        raise ValueError(f'Unexpected range response: {response.status} {response.headers.get("Content-Range")}')
                    cursor = start
                    while chunk := response.read(1024 * 1024):
                        if os.pwrite(fd, chunk, cursor) != len(chunk):
                            raise OSError('Short file write')
                        cursor += len(chunk)
                if cursor != end + 1:
                    raise OSError(f'Short archive range at {start}')
                return start
            except (OSError, ValueError) as error:
                if attempt == 4:
                    raise
                time.sleep(attempt + 1)
                print(f'Retrying {start}: {error}', flush=True)

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            tasks = [pool.submit(block, offset) for offset in starts if offset not in done]
            from concurrent.futures import as_completed
            for completed in as_completed(tasks):
                done.add(completed.result())
                record['done'] = sorted(done)
                temporary = state.with_suffix('.tmp')
                temporary.write_text(json.dumps(record))
                os.replace(temporary, state)
                if len(done) % 5 == 0 or len(done) == len(starts):
                    print(f'{args.out.name}: {len(done)}/{len(starts)} resumed ranges', flush=True)
        os.fsync(fd)
    finally:
        os.close(fd)
    args.out.with_suffix('.done').touch()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('url')
    parser.add_argument('out', type=Path)
    parser.add_argument('--workers', type=int, default=10)
    main(parser.parse_args())
