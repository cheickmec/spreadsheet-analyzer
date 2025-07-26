import json


def response(flow):
    print("----- REQUEST -----")
    print(f"URL: {flow.request.method} {flow.request.url}")
    print("Headers:")
    for k, v in flow.request.headers.items():
        print(f"  {k}: {v}")
    if flow.request.content:
        print("Body:")
        try:
            parsed_body = json.loads(flow.request.content)
            print(json.dumps(parsed_body, indent=2))
        except json.JSONDecodeError:
            print(flow.request.content.decode("utf-8", errors="ignore"))

    print("\n----- RESPONSE -----")
    print(f"Status: {flow.response.status_code}")
    print("Headers:")
    for k, v in flow.response.headers.items():
        print(f"  {k}: {v}")
    if flow.response.content:
        print("Body:")
        try:
            parsed_body = json.loads(flow.response.content)
            print(json.dumps(parsed_body, indent=2))
        except json.JSONDecodeError:
            # Handle non-JSON or gzipped content gracefully
            try:
                import gzip

                decompressed = gzip.decompress(flow.response.content)
                print(decompressed.decode("utf-8", errors="ignore"))
            except (gzip.BadGzipFile, OSError):
                print(flow.response.content.decode("utf-8", errors="ignore"))

    print("\n==================== END FLOW ====================\n")
