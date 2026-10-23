"""Run with: streamlit run app.py"""
from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import streamlit as st
from rag import Index, ingest, answer

st.set_page_config(page_title='Document Studio', page_icon='📚', layout='wide')
st.markdown('''<style>
.stApp {background: #f8fafc;}
.block-container {max-width: 1150px; padding-top: 2.5rem;}
h1 {letter-spacing: -1.5px;}
[data-testid="stSidebar"] {background: #eef2ff;}
</style>''', unsafe_allow_html=True)
st.caption('DOCUMENT STUDIO  /  LOCAL DOCUMENT ASSISTANT')
st.title('Your documents. Clear answers.')
st.write('Upload your files, ask a question, and explore the passages behind every answer.')

with st.sidebar:
    st.header('Your workspace')
    uploads = st.file_uploader('Upload documents', type=['pdf','txt'], accept_multiple_files=True)
    samples = st.checkbox('Use files from the data folder', value=not bool(uploads))
    st.divider()
    mode = st.radio('Answer mode', ['Local AI · Ollama', 'Offline · passages only'])
    semantic = st.checkbox('Semantic search', value=True, help='Downloads the embedding model on first use. Turn off for the quick lexical baseline.')
    model = st.text_input('Ollama model', 'llama3.2:3b') if mode.startswith('Local') else None
    with st.expander('Retrieval settings'):
        size = st.slider('Chunk size (characters)', 160, 2000, 500, 20)
        overlap = st.slider('Chunk overlap', 0, min(150,size-1), min(80,size-1), 1)
        k = st.slider('Passages to retrieve', 1, 10, 3)
    st.caption('Files are processed on this computer. Scanned PDFs require OCR.')

files = []
if samples:
    files.extend((p.name,p.read_bytes()) for p in sorted((Path(__file__).parent/'data').glob('*')) if p.suffix.lower() in ('.pdf','.txt'))
files.extend((Path(f.name).name,f.getvalue()) for f in uploads or [])
# Include bytes and settings: changing documents disables the old index immediately.
signature = hashlib.sha256(repr((files,size,overlap,semantic)).encode()).hexdigest()
if st.session_state.get('signature') != signature:
    for key in ('index','stats','response','hits'):
        st.session_state.pop(key,None)
    st.session_state.signature = signature

if st.button('Process documents', type='primary', disabled=not files):
    try:
        with st.spinner('Reading documents and building the search index. The first model load can take several minutes…'):
            with TemporaryDirectory() as folder:
                # Disambiguate duplicate names without letting uploads escape the folder.
                seen=set()
                for name,content in files:
                    name=Path(name).name
                    while name in seen: name='copy_'+name
                    seen.add(name)
                    (Path(folder)/name).write_bytes(content)
                chunks,stats=ingest(folder,size,overlap)
            if not chunks:
                st.error('No readable text found. Try a text-based PDF or TXT file.')
            else:
                index=Index(chunks,semantic)
                st.session_state.index=index
                st.session_state.stats=stats
    except Exception as exc:
        st.error(f'Could not process documents: {exc}')
        st.info('Check your connection for the first embedding download, or turn off Semantic search to try the offline baseline.')

if 'index' not in st.session_state:
    st.info('Choose documents on the left, then select Process documents to begin.')
    st.stop()
index=st.session_state.index
stats=st.session_state.stats
cols=st.columns(3)
for col,label,value in zip(cols,['Documents','Pages','Searchable passages'],[stats['documents'],stats['pages'],len(index.chunks)]):
    col.metric(label,value)
if stats['empty_pages']: st.warning(f"Skipped {stats['empty_pages']} empty pages.")
for error in stats['errors']: st.warning(error)
with st.expander('Inspect sample chunks'):
    for chunk in index.chunks[:3]:
        st.caption(f"{chunk['source']} · Page {chunk['page']} · Category: {chunk['category']}")
        st.text(chunk['text'])

st.subheader('Ask your documents')
category=st.selectbox('Limit to category', ['All categories']+sorted({c['category'] for c in index.chunks}))
with st.form('question'):
    question=st.text_input('Your question', placeholder='What does the document say about machine learning?')
    submit=st.form_submit_button('Find an answer',type='primary')
if submit:
    st.session_state.pop('response',None)
    st.session_state.pop('hits',None)
    if not question.strip(): st.warning('Enter a question first.')
    elif model is not None and not model.strip(): st.warning('Enter an Ollama model name.')
    else:
        try:
            with st.spinner('Finding evidence and preparing your answer…'):
                hits=index.search(question,k,None if category=='All categories' else category)
                st.session_state.hits=hits
                st.session_state.response=answer(question,hits,model)
                st.session_state.asked=question
        except Exception as exc:
            st.error(f'Answer generation failed: {exc}')
            st.info('For Local AI, open Ollama and ensure the selected model is downloaded. You can also choose Offline mode.')
if 'response' in st.session_state:
    st.subheader('Answer')
    st.caption(st.session_state.asked)
    st.write(st.session_state.response)
    st.download_button('Download answer',st.session_state.response,file_name='answer.txt')
if 'hits' in st.session_state:
    st.subheader('Sources & retrieved context')
    st.caption('These passages were provided to the model. Check them to verify its claims.')
    for i,hit in enumerate(st.session_state.hits,1):
        with st.expander(f"[{i}] {hit['source']} · Page {hit['page']}", expanded=i==1):
            st.caption(f"Category: {hit['category']} · Similarity: {hit['score']:.3f}")
            st.text(hit['text'])
