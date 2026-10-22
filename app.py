import streamlit as st
import PyPDF2
import faiss
from sentence_transformers import SentenceTransformer
import requests
# ---------------- AI ANSWER ----------------
def generate_answer(context, question):
    prompt = f"""You are an AI College Study Assistant.

Use ONLY the information in the provided PDF context to answer
the student's question.

Do not repeat the prompt or the student's question.
Do not write labels like "Student's question" or "Answer:".
Give only the actual answer in simple English.
If the answer is not in the context, say:
"I could not find this answer in the uploaded PDF."

PDF Context:
{context}

Question: {question}

Response:"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2:1b",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2
            }
        },
        timeout=120
    )

    response.raise_for_status()
    answer = response.json().get("response", "").strip()

    return answer or "Sorry, I could not generate an answer. Please try again."
    
    response.raise_for_status()
    answer = response.json().get("response", "").strip()

    return answer or "Sorry, I could not generate an answer. Please try again."


# ---------------- QUIZ GENERATOR FUNCTION ----------------
def generate_quiz(context, num_questions=5):
    import json

    prompt = f"""
Create exactly {num_questions} MCQs using ONLY the PDF context.
Use simple English. Each question must have four options and one
correct answer. Return only valid JSON, without markdown.

Format:
{{
  "questions": [
    {{
      "question": "Question text",
      "options": {{
        "A": "Option A",
        "B": "Option B",
        "C": "Option C",
        "D": "Option D"
      }},
      "answer": "A",
      "explanation": "Short explanation"
    }}
  ]
}}

PDF context:
{context}
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2:1b",
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.2}
        },
        timeout=180
    )

    response.raise_for_status()
    result = json.loads(response.json()["response"])

    if isinstance(result, dict):
        result = result.get("questions", [])

    if not isinstance(result, list):
        raise ValueError("Invalid quiz format returned by AI.")

    valid_questions = []

    for item in result:
        if not isinstance(item, dict):
            continue

        options = item.get("options", {})
        answer = str(item.get("answer", "")).strip().upper()

        if (
            item.get("question")
            and isinstance(options, dict)
            and all(k in options for k in ["A", "B", "C", "D"])
            and answer in ["A", "B", "C", "D"]
        ):
            valid_questions.append(item)

    if not valid_questions:
        raise ValueError("No valid MCQs generated. Try again.")

    return valid_questions[:num_questions]


# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="AI College Study Assistant",
    page_icon="📚",
    layout="wide"
)
# Quiz session state initialization
if "quiz_questions" not in st.session_state:
    st.session_state.quiz_questions = []

if "quiz_answers" not in st.session_state:
    st.session_state.quiz_answers = {}

if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False




# ---------------- DARK THEME ----------------
st.markdown("""
<style>
.stApp {
    background-color: #212121;
    color: #ECECEC;
}

.block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3, p, label {
    color: #ECECEC !important;
}

.stTextInput input {
    background-color: #303030;
    color: #FFFFFF;
    border: 1px solid #505050;
    border-radius: 12px;
}

[data-testid="stFileUploader"] {
    background-color: #2B2B2B;
    border: 1px solid #454545;
    border-radius: 12px;
    padding: 12px;
}

.stButton button {
    background-color: #10A37F;
    color: white;
    border: none;
    border-radius: 10px;
}

.stTextArea textarea {
    background-color: #303030;
    color: #ECECEC;
    border-radius: 10px;
}

[data-testid="stChatMessage"] {
    background-color: #292929;
    border: 1px solid #383838;
    border-radius: 14px;
    padding: 12px;
}
</style>
""", unsafe_allow_html=True)


# ---------------- APP TITLE ----------------
st.title("📚 AI College Study Assistant")
st.write("Upload your college notes and learn smarter!")


# ---------------- LOAD EMBEDDING MODEL ----------------
@st.cache_resource
def load_model():
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


# ---------------- PDF UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload your notes PDF",
    type=["pdf"]
)


if uploaded_file is not None:

    try:
        reader = PyPDF2.PdfReader(uploaded_file)
    except Exception as e:
        st.error(f"Could not read the PDF: {e}")
        st.stop()

    page_texts = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_content = page.extract_text() or ""

        page_texts.append({
            "page": page_number,
            "text": page_content
        })

    full_text = "\n".join(
        page["text"] for page in page_texts
    )

    st.success("PDF uploaded successfully!")
    st.write("Total pages:", len(reader.pages))

    if not full_text.strip():
        st.warning(
            "PDF se text nahi mila. Ho sakta hai PDF scanned ho."
        )
        st.stop()

    with st.expander("📄 View Extracted PDF Text"):
        st.text_area(
            "PDF content",
            full_text,
            height=200
        )

    # ---------------- CREATE CHUNKS ----------------
    chunk_size = 800
    overlap = 100

    chunks = []
    chunk_pages = []

    for page_info in page_texts:
        page_text = page_info["text"]
        page_number = page_info["page"]

        step_size = chunk_size - overlap

        for start in range(0, len(page_text), step_size):
            chunk = page_text[start:start + chunk_size]

            if chunk.strip():
                chunks.append(chunk)
                chunk_pages.append(page_number)

    if not chunks:
        st.warning("PDF se usable text chunks nahi ban paaye.")
        st.stop()

    st.subheader("🧩 PDF Text Chunks")
    st.write("Total chunks:", len(chunks))

    with st.expander("Preview a text chunk"):
        selected_chunk = st.selectbox(
            "Choose a chunk",
            range(len(chunks)),
            format_func=lambda i: f"Chunk {i + 1}"
        )

        st.text_area(
            "Chunk content",
            chunks[selected_chunk],
            height=150
        )

    # ---------------- SMART SEARCH ----------------
    st.subheader("🔎 Ask Your Notes")

    question = st.text_input(
        "Ask a question about your uploaded PDF",
        placeholder="Example: Explain the main topic in simple English"
    )

    if question.strip():

        try:
            with st.spinner("Searching your notes..."):

                model = load_model()

                chunk_embeddings = model.encode(
                    chunks,
                    normalize_embeddings=True,
                    convert_to_numpy=True
                ).astype("float32")

                index = faiss.IndexFlatIP(
                    chunk_embeddings.shape[1]
                )

                index.add(chunk_embeddings)

                question_embedding = model.encode(
                    [question],
                    normalize_embeddings=True,
                    convert_to_numpy=True
                ).astype("float32")

                scores, indices = index.search(
                    question_embedding,
                    min(3, len(chunks))
                )

            # Keep the best matching chunks and their page numbers
            relevant_chunks = []
            relevant_pages = []

            for idx in indices[0]:
                if 0 <= idx < len(chunks):
                    relevant_chunks.append(chunks[idx])
                    relevant_pages.append(chunk_pages[idx])

            context = "\n\n".join(relevant_chunks)

            # ---------------- RELEVANT SOURCES ----------------
            with st.expander("📚 View Relevant Notes and Sources"):
                for rank, idx in enumerate(indices[0]):
                    if 0 <= idx < len(chunks):
                        st.markdown(
                            f"**Source {rank + 1} — PDF Page "
                            f"{chunk_pages[idx]}** "
                            f"(Similarity: {scores[0][rank]:.3f})"
                        )
                        st.write(chunks[idx])

            # ---------------- GENERATE ANSWER ----------------
            st.subheader("🤖 AI Answer")

            st.chat_message("user").write(question)

            with st.spinner("AI is preparing your answer..."):
                answer = generate_answer(context, question)

            st.chat_message("assistant").write(answer)

            unique_pages = sorted(set(relevant_pages))
            if unique_pages:
                pages_text = ", ".join(
                    str(page) for page in unique_pages
                )
                st.caption(
                    f"📖 Sources: Uploaded PDF, page(s) {pages_text}"
                )

        except requests.exceptions.ConnectionError:
            st.error(
                "Ollama connect nahi ho raha. "
                "Ek terminal mein `ollama run llama3.2:1b` chala kar check karo."
            )

        except requests.exceptions.Timeout:
            st.error(
                "AI response mein zyada time lag raha hai. "
                "Dobara try karo."
            )

        except requests.exceptions.RequestException as e:
            st.error(f"Ollama request failed: {e}")

        except Exception as e:
            st.error(f"Something went wrong: {e}")

# ---------------- QUIZ GENERATOR UI ----------------
st.divider()
st.header("📝 PDF Quiz Generator")

num_questions = st.selectbox(
    "Number of MCQs",
    [3, 5, 10],
    index=1
)

if st.button("🎯 Generate Quiz"):
    if not chunks:
        st.warning("Please upload a PDF first.")
    else:
        quiz_context = "\n\n".join(chunks[:20])

        with st.spinner("Generating MCQs from your PDF..."):
            try:
                st.session_state.quiz_questions = generate_quiz(
                    quiz_context,
                    num_questions
                )
                st.session_state.quiz_submitted = False
                st.session_state.quiz_answers = {}
            except Exception as e:
                st.error(f"Quiz generation failed: {e}")

if st.session_state.get("quiz_questions"):
    quiz_questions = st.session_state.quiz_questions

    st.subheader("Choose your answers")

    with st.form("quiz_form"):
        for i, q in enumerate(quiz_questions):
            st.markdown(f"**Q{i + 1}. {q['question']}**")

            selected = st.radio(
                f"Choose an option for Q{i + 1}",
                ["A", "B", "C", "D"],
                format_func=lambda key, opts=q["options"]: (
                    f"{key}. {opts[key]}"
                ),
                key=f"quiz_q_{i}",
                index=None
            )

            st.session_state.quiz_answers[i] = selected

        submitted = st.form_submit_button("✅ Submit Quiz")

    if submitted:
        st.session_state.quiz_submitted = True

    if st.session_state.get("quiz_submitted", False):
        score = 0

        for i, q in enumerate(quiz_questions):
            user_answer = st.session_state.quiz_answers.get(i)

            if user_answer == q["answer"].strip().upper():
                score += 1

        st.success(
            f"Your score: {score}/{len(quiz_questions)}"
        )

        st.subheader("📖 Answer Review")

        for i, q in enumerate(quiz_questions):
            correct = q["answer"].strip().upper()
            user_answer = st.session_state.quiz_answers.get(i)

            st.markdown(f"**Q{i + 1}. {q['question']}**")
            st.write(f"Your answer: {user_answer or 'Not answered'}")
            st.write(f"Correct answer: {correct}. {q['options'][correct]}")
            st.write(f"Explanation: {q.get('explanation', 'No explanation provided.')}")
            st.divider()
