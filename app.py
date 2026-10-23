"""Minimal document question-answering interface."""
import hashlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import streamlit as st
from rag import Index, ingest, answer

st.set_page_config(page_title='Ask your documents', page_icon='📄', layout='centered', initial_sidebar_state='collapsed')
st.markdown('''<style>
.stApp{background:#f6f8f7;color:#182c25;color-scheme:light}
.block-container{max-width:860px;padding-top:3.5rem;padding-bottom:4rem}
h1{letter-spacing:-1.6px;font-size:2.7rem!important;font-weight:700!important;color:#163a2c!important}
h2,h3{color:#214c3b!important;font-size:1.2rem!important;font-weight:650!important}
[data-testid="stHeader"]{background:transparent}
[data-testid="stSidebar"]{background:#edf3ef;color:#182c25}
[data-testid="stSidebar"] p,[data-testid="stSidebar"] label{color:#263e33!important}
[data-testid="stCaptionContainer"] p{color:#5a6c63!important;line-height:1.6}
[data-testid="stFileUploaderDropzone"]{background:#202d28!important;border:1px dashed #729783!important;border-radius:16px!important;padding:24px!important}
[data-testid="stFileUploaderDropzone"] p,[data-testid="stFileUploaderDropzone"] small,[data-testid="stFileUploaderDropzone"] span{color:#f1f7f3!important}
[data-testid="stFileUploaderDropzone"] button{background:#345c48!important;border:1px solid #85ad94!important;color:#fff!important;border-radius:10px!important}
[data-testid="stFileUploaderDropzone"] button p,[data-testid="stFileUploaderDropzone"] button span,[data-testid="stFileUploaderDropzone"] button svg{color:#fff!important;fill:currentColor}
[data-testid="stFileUploaderFile"]{background:#fff!important;color:#243d30!important;border-radius:8px;padding:8px}
[data-testid="stFileUploaderFile"] *{color:#243d30!important}
[data-testid="stTextInput"] input{background:#fff!important;color:#182c25!important;caret-color:#182c25!important;border:1px solid #cad8cf!important;border-radius:10px!important}
[data-testid="stTextInput"] input::placeholder{color:#74867a!important}
[data-testid="stTextInput"] label p,[data-testid="stFileUploader"] label p{color:#314d3d!important}
.stButton>button,.stDownloadButton>button{border-radius:10px!important;min-height:44px;font-weight:600}
.stButton>button[kind="primary"]{background:#276749!important;border-color:#276749!important;color:white!important}
.stButton>button[kind="primary"] p{color:white!important}
.stButton>button:hover{border-color:#63a47e!important}
[data-testid="stVerticalBlockBorderWrapper"]>div{border-color:#dce6de!important;border-radius:14px!important;background:#fff}
[data-testid="stText"]{white-space:pre-wrap;color:#263e33!important;font-family:inherit!important;line-height:1.65!important}
[data-testid="stForm"]{background:white;border:1px solid #dce6de!important;border-radius:14px;padding:20px!important}
[data-testid="stAlert"]{border-radius:12px}
hr{border-color:#dce6de!important}
</style>''', unsafe_allow_html=True)
st.caption('DOCUMENT ASSISTANT')
st.title('Ask your documents')
st.write('Turn your documents into clear answers, backed by sources.')
hosted=bool(os.getenv('RENDER'))
cloud=bool(os.getenv('OLLAMA_API_KEY'))
with st.sidebar:
    st.header('Settings')
    st.caption('The default search works without a model download.')
    semantic=st.checkbox('Use semantic search',value=False,help='Requires a downloaded embedding model and enough server memory.')
    local_ai=st.checkbox('Use local Ollama',value=False,disabled=hosted)
    if hosted and not cloud: st.caption('Ollama Cloud is waiting for its server API key. Search returns original passages until connected.')
    if cloud: st.caption('AI answers powered by Ollama Cloud.')
    model=os.getenv('OLLAMA_MODEL','gemma4:31b') if cloud else (st.text_input('Model name','llama3.2:3b') if local_ai else None)
    k=st.slider('Number of passages',1,5,3)
    size=st.slider('Passage length',200,1500,500,100)
    overlap=st.slider('Passage overlap',0,150,80,10)
    st.caption('Categories are taken from the first part of each filename, before an underscore.')

st.subheader('1. Add your documents')
uploads=st.file_uploader('Choose PDF or text files',type=['pdf','txt'],accept_multiple_files=True,help='Use PDFs with selectable text. Scanned pages need OCR first.')
st.caption('Files are processed on the server for this session. When you ask a question, selected passages are sent to Ollama Cloud for an AI answer.' if cloud else 'Files are processed on the server for this session.')
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
    submitted=st.form_submit_button('Get answer' if cloud or local_ai else 'Search documents',type='primary')
if submitted:
    for key in ('result','hits','asked','answer_mode'): st.session_state.pop(key,None)
    if not question.strip(): st.warning('Please enter a question.')
    elif (local_ai or cloud) and not model.strip(): st.warning('Please enter a model name in Settings.')
    else:
        try:
            with st.spinner('Looking through your documents…'):
                hits=index.search(question,k,category)
                st.session_state.hits=hits; st.session_state.asked=question
                st.session_state.answer_mode=bool(local_ai or cloud)
                if local_ai or cloud: st.session_state.result=answer(question,hits,model)
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
            st.write(hit['text'])
    if hits:
        export=(st.session_state.get('result','')+'\n\n'+'\n\n'.join(f"[{i}] {h['source']}, page {h['page']}\n{h['text']}" for i,h in enumerate(hits,1))).strip()
        st.download_button('Save results',export,file_name='document-answer.txt')
