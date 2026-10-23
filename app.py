"""Minimal document question-answering interface."""
import hashlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import streamlit as st
from rag import Index, ingest, answer

st.set_page_config(page_title='Ask your documents', page_icon='📄', layout='centered', initial_sidebar_state='collapsed')
st.markdown('''<style>
.stApp{background:#fafaf9;color:#202521}
.block-container{max-width:820px;padding-top:3rem;padding-bottom:4rem}
h1{letter-spacing:-1.5px;font-size:2.6rem!important;font-weight:650!important}
h2{font-size:1.15rem!important;margin-top:1rem!important}
[data-testid="stHeader"]{background:transparent}
[data-testid="stMetric"]{background:white;border:1px solid #e7e8e5;border-radius:10px;padding:12px}
.stButton>button[kind="primary"]{background:#276749;border-color:#276749}
</style>''', unsafe_allow_html=True)
st.title('Ask your documents')
st.write('Find answers in your PDFs, with sources you can check.')
hosted=bool(os.getenv('RENDER'))
with st.sidebar:
    st.header('Settings')
    st.caption('The default search works without a model download.')
    semantic=st.checkbox('Use semantic search',value=False,help='Requires a downloaded embedding model and enough server memory.')
    local_ai=st.checkbox('Use local Ollama',value=False,disabled=hosted)
    if hosted: st.caption('AI writing is not connected on this website. Search returns original document passages.')
    model=st.text_input('Model name','llama3.2:3b') if local_ai else None
    k=st.slider('Number of passages',1,5,3)
    size=st.slider('Passage length',200,1500,500,100)
    overlap=st.slider('Passage overlap',0,150,80,10)
    st.caption('Categories are taken from the first part of each filename, before an underscore.')

st.subheader('1. Add your documents')
uploads=st.file_uploader('Choose PDF or text files',type=['pdf','txt'],accept_multiple_files=True,help='Use PDFs with selectable text. Scanned pages need OCR first.')
st.caption('Files are processed on the server for this session. Upload only documents you are comfortable processing here.')
files=[(Path(f.name).name,f.getvalue()) for f in uploads or []]
sig=hashlib.sha256(repr((files,semantic,size,overlap)).encode()).hexdigest()
if st.session_state.get('signature')!=sig:
    for key in ('index','stats','result','hits','asked','answer_mode'): st.session_state.pop(key,None)
    st.session_state.signature=sig
if files and 'index' not in st.session_state:
    if st.button('Prepare documents',type='primary'):
        try:
            with st.spinner('Preparing your documents. First-time semantic search may take a few minutes…'):
                with TemporaryDirectory() as folder:
                    seen=set()
                    for name,content in files:
                        while name in seen: name='copy_'+name
                        seen.add(name); (Path(folder)/name).write_bytes(content)
                    chunks,stats=ingest(folder,size,overlap)
                if not chunks: st.error('No readable text found. Please use a PDF with selectable text or a TXT file.')
                else:
                    st.session_state.index=Index(chunks,semantic)
                    st.session_state.stats=stats
        except Exception as exc:
            st.error('We could not prepare these documents. Try a smaller file or turn off semantic search in Settings.')
            with st.expander('Error details'): st.text(str(exc))
if 'index' not in st.session_state:
    if not files: st.info('Add a document above to get started.')
    st.stop()
index=st.session_state.index; stats=st.session_state.stats
st.success(f"Ready · {stats['documents']} document(s) · {stats['pages']} page(s)")
if stats['empty_pages']: st.warning(f"{stats['empty_pages']} page(s) had no readable text and were skipped.")
for error in stats['errors']: st.warning(error)
st.subheader('2. Ask a question')
categories=sorted({c['category'] for c in index.chunks})
category=None
if len(categories)>1:
    with st.expander('Search a specific category'):
        selected=st.selectbox('Category',['All documents']+categories)
        category=None if selected=='All documents' else selected
with st.form('question'):
    question=st.text_input('What would you like to know?',placeholder='For example: What are the main findings?')
    submitted=st.form_submit_button('Search documents',type='primary')
if submitted:
    for key in ('result','hits','asked','answer_mode'): st.session_state.pop(key,None)
    if not question.strip(): st.warning('Please enter a question.')
    elif local_ai and not model.strip(): st.warning('Please enter a model name in Settings.')
    else:
        try:
            with st.spinner('Looking through your documents…'):
                hits=index.search(question,k,category)
                st.session_state.hits=hits; st.session_state.asked=question
                st.session_state.answer_mode=bool(local_ai)
                if local_ai: st.session_state.result=answer(question,hits,model)
        except Exception as exc:
            st.error('The answer could not be generated. Your retrieved passages are shown below if available.')
            with st.expander('Error details'): st.text(str(exc))
if 'hits' in st.session_state:
    st.divider()
    st.subheader('3. Read the results')
    st.caption(st.session_state.asked)
    hits=st.session_state.hits
    if 'result' in st.session_state: st.write(st.session_state.result)
    elif hits: st.write('Here are the closest passages from your documents. Check whether they answer your question.')
    else: st.info('No passages were found in the selected documents.')
    if hits and not st.session_state.get('answer_mode'): st.caption('Original passages · AI-written answers are not enabled. A close match may still be unrelated to your question.')
    for i,hit in enumerate(hits,1):
        with st.container(border=True):
            st.markdown(f'**Source {i}**')
            st.caption(f"{hit['source']} · Page {hit['page']}")
            st.text(hit['text'])
    if hits:
        export=(st.session_state.get('result','')+'\n\n'+'\n\n'.join(f"[{i}] {h['source']}, page {h['page']}\n{h['text']}" for i,h in enumerate(hits,1))).strip()
        st.download_button('Save results',export,file_name='document-answer.txt')
