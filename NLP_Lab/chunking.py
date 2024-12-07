def chunk_document(doc: str, desired_chunk_size: int = 500, max_chunk_size: int = 500):
    chunk = ''
    for line in doc.splitlines():
        chunk += line + '\n'
        if len(chunk) >= desired_chunk_size:
            yield chunk[:max_chunk_size]
            chunk = ''
    if chunk:
        yield chunk


def chunk_documents(docs: [str], desired_chunk_size: int = 500, max_chunk_size: int = 500):
    chunks = []
    for doc in docs:
        chunks += list(chunk_document(doc))
    return chunks
