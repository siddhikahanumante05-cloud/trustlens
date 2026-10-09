"""Test tool: streams a local .webm file to TrustLens WebSocket in 1-second fragments."""
import asyncio
import argparse
import json
import os
import sys
import time
import websockets


async def stream_file(file_path: str, ws_url: str, chunk_interval_sec: float = 1.0, consent: bool = True):
    """Stream a local webm file in 1-second fragments to the analysis websocket."""
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        return 1

    file_size = os.path.getsize(file_path)
    print(f"Streaming {file_path} ({file_size} bytes) to {ws_url}...")

    # For 1-second chunks, slice file proportionally or in fixed blocks
    # WebM video bitrate typically ~500 kbps - 2 Mbps (60 KB - 250 KB / s)
    total_est_seconds = 20.0
    chunk_size = max(4096, int(file_size / total_est_seconds))

    async with websockets.connect(ws_url) as ws:
        # 1. Send start handshake with caller consent
        start_msg = {
            "type": "start",
            "client_time": int(time.time() * 1000),
            "consent": consent,
        }
        await ws.send(json.dumps(start_msg))
        print("Sent 'start' handshake (consent={})".format(consent))

        # Background listener for server events
        async def listen_events():
            try:
                async for message in ws:
                    try:
                        data = json.loads(message)
                        mtype = data.get("type")
                        if mtype == "check":
                            print(f"[CHECK] {data.get('id')}: {data.get('state')}")
                        elif mtype == "finding":
                            print(f"[FINDING] ({data.get('severity')}) {data.get('title')}: {data.get('detail')}")
                        elif mtype == "risk":
                            print(f"[RISK] {data.get('value'):.2f}")
                        elif mtype == "action":
                            print(f"[ACTION] {data.get('text')}")
                        elif mtype == "summary":
                            print(f"[SUMMARY] {data.get('text')}")
                        elif mtype == "error":
                            print(f"[ERROR] {data.get('code')}: {data.get('message')}")
                    except Exception:
                        pass
            except asyncio.CancelledError:
                pass
            except Exception as e:
                print(f"Listener closed: {e}")

        listener_task = asyncio.create_task(listen_events())

        # 2. Feed binary chunks
        with open(file_path, "rb") as f:
            bytes_sent = 0
            chunk_idx = 0
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                await ws.send(chunk)
                bytes_sent += len(chunk)
                chunk_idx += 1
                print(f"Sent chunk {chunk_idx}: {len(chunk)} bytes ({bytes_sent}/{file_size})")
                await asyncio.sleep(chunk_interval_sec)

        print("Finished sending all chunks. Waiting 5s for remaining events...")
        await asyncio.sleep(5.0)
        listener_task.cancel()

    print("Stream session completed.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Stream WebM file to TrustLens WebSocket")
    parser.add_argument("--file", required=True, help="Path to .webm file")
    parser.add_argument("--url", default="ws://localhost:8000/ws/analyze", help="WebSocket URL")
    parser.add_argument("--interval", type=float, default=1.0, help="Interval between chunks in seconds")
    parser.add_argument("--no-consent", action="store_true", help="Omit caller consent (test rejection)")
    args = parser.parse_args()

    asyncio.run(
        stream_file(
            file_path=args.file,
            ws_url=args.url,
            chunk_interval_sec=args.interval,
            consent=not args.no_consent,
        )
    )


if __name__ == "__main__":
    main()
