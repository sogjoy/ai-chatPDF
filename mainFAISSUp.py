# from dotenv import load_dotenv
# load_dotenv()
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import streamlit as st
import tempfile
import os

# 제목
st.title("ChatPDF")
st.write("---")

# 파일 업로드
uploaded_file = st.file_uploader("PDF 파일을 올려 주세요", type=["pdf"])
st.write("---")

def pdf_to_document(uploaded_file):
    # PDF 데이터를 임시 파일에 저장
    pdf_bytes = uploaded_file.getvalue()
    temp_dir = tempfile.TemporaryDirectory()
    temp_pdf_path = os.path.join(temp_dir.name, uploaded_file.name)

    with open(temp_pdf_path, "wb") as f:
        f.write(pdf_bytes)

    # 임시 파일 경로로 PyPDFLoader 로드
    loader = PyPDFLoader(temp_pdf_path)
    pages = loader.load_and_split()
    temp_dir.cleanup()  # 임시 디렉토리 정리
    return pages

# 업로드되면 동작하는 코드
if uploaded_file is not None:
    with st.spinner('PDF 처리 중...'):
        pages = pdf_to_document(uploaded_file)

        # Split
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False,
        )

        texts = text_splitter.split_documents(pages)

        # Embedding
        embeddings_model = OpenAIEmbeddings()

        # FAISS 벡터 저장소 생성
        db = FAISS.from_documents(texts, embeddings_model)

        # 질문 처리
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        
        # 프롬프트 템플릿 정의
        prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}
        Answer:"""
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        
        # Retrieval QA 체인 생성
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
    
    st.success('PDF 처리 완료!')
    
    # Streamlit을 통한 질문 입력 및 처리
    st.header("PDF에게 질문해보세요!!")
    question = st.text_input("질문을 입력하세요")

    if st.button("질문하기"):
        try:
            with st.spinner('답변 생성 중...'):
                # 결과 가져오기 (invoke 메서드 사용)
                result = qa_chain.invoke({"query": question})
            st.write(result['result'])
        except Exception as e:
            st.error(f"오류 발생: {str(e)}")
else:
    st.info('PDF 파일을 업로드해주세요.')

# OpenAI API 키 확인 (디버깅용, 실제 배포 시 제거)
if not os.getenv('OPENAI_API_KEY'):
    st.warning('OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.')