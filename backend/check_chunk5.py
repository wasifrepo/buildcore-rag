import chromadb
client = chromadb.PersistentClient(path='./data/chroma')
col = client.get_collection('buildcore')
results = col.get(where={'document_id': 'MAINT-FLT-03-forklift-toyota-8fgf25'}, include=['documents', 'metadatas'])
for i, (doc, meta) in enumerate(zip(results['documents'], results['metadatas'])):
    title = meta.get('section_title', '')
    if 'PRE-START' in title.upper():
        print('Length:', len(doc))
        print(doc)
