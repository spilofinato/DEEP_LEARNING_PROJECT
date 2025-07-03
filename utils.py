# Create summarization utilities module
import textwrap
from collections import defaultdict
from datetime import datetime as dt
from http.client import responses
from time import sleep

import ollama
import torch
from google import genai
from openai import OpenAI
from bs4 import BeautifulSoup
from markdown_it import MarkdownIt
import nltk
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
import re
from difflib import SequenceMatcher

nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize, sent_tokenize

import spacy
nlp = spacy.load("it_core_news_sm")

from ispin.ISPIn import ISPIn

import api_keys

def is_long_progress(progress, msg_threshold=30, words_threshold=1500):
    return progress["message_count"] >= msg_threshold or progress["total_words"] >= words_threshold

def count_words(text):
    tokens = word_tokenize(text, language='italian')
    return len([t for t in tokens if t.isalpha()])

def split_sentences(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

def strip_markdown_and_html(text):
    # Step 1: remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()

    # Step 2: remove common Markdown symbols
    text = re.sub(r"\*{1,2}|_{1,2}|`{1,3}|#{1,6}|\[.*?\]\(.*?\)", "", text)

    # Step 3: collapse excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

def clean_sentences(sentences):
    cleaned = []
    for s in sentences:
        s = s.strip()
        if len(s) < 6 or len(s) > 300:
            continue
        if re.fullmatch(r"[-*_.:\s]+", s):
            continue
        if re.search(r"\b(?:original length|estimated input tokens|query generated|dataframe)\b", s.lower()):
            continue
        if s.count(s.split()[0]) > 5:
            continue  # repeated same word
        if re.search(r"\b(pacq|xls|nan|ordine-|pdf|20\d\d[-/])", s.lower()):
            continue
        cleaned.append(s)
    return cleaned

def strip_summary_metadata(summary):
    return re.split(r"(?=---\s*Original length:)", summary)[0].strip()

def is_question_garbage(q):
    if len(q) < 10 or len(q) > 300:
        return True
    if q.count("?") > 5 or q.count(q.split()[0]) > 4:
        return True
    if re.search(r"\b(query|token|generated in|length)\b", q.lower()):
        return True
    return False

def remove_near_duplicates(lst, threshold=0.85):
    seen = []
    for s in lst:
        if all(SequenceMatcher(None, s, prev).ratio() < threshold for prev in seen):
            seen.append(s)
    return seen

def is_similar(a, b, threshold=0.85):
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio() > threshold

def build_chronological_text_from_progress(progress):
    # Build a chronological text from messages, tokenizing by author and content

    messages = progress.get("messages", [])

    if not messages:
        return ""

    # Sort messages by message_id to ensure chronological order
    messages = sorted(messages, key=lambda x: x.get("message_id", 0))

    # Join messages into a single string with author and content

    text = textwrap.dedent(
            f"""Titolo: {progress['subject']} - Creato il: {progress['created_at']} - ID progetto: {progress['progress_id']}
                Descrizione:{progress['description']}
                Numero di messaggi: {progress['message_count']}
                
                Messaggi:
            """).strip()

    for m in messages:
        m["content"] = m.get("content", "")

        # Cerco in messages data e autore del messaggio a cui si sta rispondendo

        if m.get("answer_to_message_id", 0) > 0:
            # Find the message being replied to
            answer_to_message = next((msg for msg in messages if msg['message_id'] == m['answer_to_message_id']), None)
            if answer_to_message:
                m['answered_message_author'] = answer_to_message['author']
                m['answered_message_timestamp'] = answer_to_message['timestamp']
            else:
                m['answered_message_author'] = ""
                m['answered_message_timestamp'] = ""

        text = textwrap.dedent(
            f"""{text}
            Autore: {m['author']} - Data: {m['timestamp']} {f"- In risposta al messaggio di: {m.get('answered_message_author', '')} scritto il {m.get('answered_message_timestamp', '')}" if m.get('answer_to_message_id', 0) > 0 else ''}
            Contenuto:
            {m['content']}
            {f"Allegato: {m['attachment']}" if m.get("attachment") else ""}
            """)

    return text.strip()

def split_progress_by_length(messages, words_threshold=1500):
    chunks = []
    current_chunk = []
    current_length = 0
    for msg in messages:
        msg_length = count_words(msg.get("content", ""))
        if current_length + msg_length > words_threshold and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_length = 0
        current_chunk.append(msg)
        current_length += msg_length
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def remove_markdown(md_text):
    html = MarkdownIt().render(md_text)
    return BeautifulSoup(html, "html.parser").get_text()

def build_summarization_prompt(text, mode="full"):
    text_for_prompt = text

    if isinstance(text_for_prompt, list):
        rewrite_text_for_prompt = ""
        for text in text_for_prompt:
            if not isinstance(text, str):
                raise ValueError("All elements in the list must be strings.")
            rewrite_text_for_prompt = textwrap.dedent(f"""
                {rewrite_text_for_prompt}
                
                {remove_markdown(text)}
                """)
        text_for_prompt = rewrite_text_for_prompt.strip()

    max_length = max(200, min(1500, count_words(text_for_prompt) // 2))

    print(f"Max length for summarization: {max_length} words.")

    if mode == "full":
        prompt = textwrap.dedent(f"""
        Istruzioni:
            Riassumi il seguente dettaglio di progetto in formato cronologico
            Usa un formato ad elenco puntato per gli eventi principali.
            Evidenzia le persone coinvolte e le loro azioni.
            Evidenzia la presenza di allegati ai messaggi.
            Scrivi esclusivamente in italiano.
            Non includere introduzioni né conclusioni.
            Lunghezza massima del riassunto da generare: {max_length} parole.

        Dettaglio del progetto:
        {text_for_prompt}
        """)
    elif mode == "hierarchical":
        prompt = textwrap.dedent(f"""
        Istruzioni:
            Unisci i testi forniti in un unico riassunto coerente in formato cronologico.
            Usa un formato ad elenco puntato.
            Evidenzia le persone coinvolte e le loro azioni.
            Evidenzia la presenza di allegati ai messaggi.
            Scrivi esclusivamente in italiano.
            Non aggiungere introduzioni né conclusioni.
            Lunghezza massima del riassunto da generare: {max_length} parole.
            Evita ripetizioni.
            
        Testo da unire:
        {text_for_prompt}
        """)
    else:
        raise ValueError("Unsupported prompt mode.")

    # print(prompt.strip())
    return prompt.strip()

def split_progress_by_timeframe(progress, timeframe=1.0):
    if timeframe not in {1.0, 0.5, 0.25}:
        raise ValueError("Unsupported timeframe. Use 1.0 for year, 0.5 for semester, or 0.25 for quarter.")

    buckets = defaultdict(list)
    last_key = None

    for msg in progress["messages"]:
        try:
            ts = dt.strptime(msg["timestamp"], "%Y-%m-%d %H:%M:%S")
            year = ts.year
            if timeframe == 1.0:
                key = f"{year}"
            elif timeframe == 0.5:
                semester = 1 if ts.month <= 6 else 2
                key = f"{year}-S{semester}"
            elif timeframe == 0.25:
                quarter = (ts.month - 1) // 3 + 1
                key = f"{year}-Q{quarter}"
            else:
                raise ValueError("Unsupported timeframe. Use 1.0 for year, 0.5 for semester, or 0.25 for quarter.")
            last_key = key
        except ValueError:
            if last_key is None:
                continue  # scarta il messaggio se non abbiamo ancora un bucket valido
            key = last_key

        buckets[key].append(msg)

    # Restituisci i bucket ordinati cronologicamente
    return [buckets[key] for key in sorted(buckets.keys())]

def summarize_with_ollama(prompt):
    response = None

    while response is None:
        try:
            # Use the ollama client to generate a summary
            response = ollama.generate(
                model="gemma3:4b",
                prompt=prompt,
            )
        except Exception as e:
            print(f"Error generating content with Ollama: {e}, sleeping for 10 seconds before retrying...")
            sleep(10)
            response = None

    print("Summary length:", count_words(response["response"]), "words.")

    return response["response"]

def summarize_with_gemini(prompt):
    google_client = genai.Client(api_key=api_keys.GOOGLE_API_KEY)

    response = None

    while response is None:
        try:
            response = google_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
        except Exception as e:
            print(f"Error generating content with Gemini: {e}, sleeping for 10 seconds before retrying...")
            sleep(10)
            response = None

    print("Summary length:", count_words(response.text), "words.")

    return response.text

def summarize_with_gpt(prompt):
    client = OpenAI(api_key=api_keys.OPENAI_API_KEY)

    response = None

    while response is None:
        try:
            response = client.responses.create(
                model="gpt-4.1",
                input=prompt,
            )
        except Exception as e:
            print(f"Error generating content with GPT: {e}")
            sleep(10)
            response = None

    print("Summary length:", count_words(response.output_text), "words.")

    return response.output_text

def summarize_chunked_progress(messages, progress_for_summary, llm, mode):
    summaries = []
    for chunk in messages:
        progress_for_summary["messages"] = chunk
        text = build_chronological_text_from_progress(progress_for_summary)
        prompt = build_summarization_prompt(text, mode=mode)
        if llm == "ollama":
            summaries.append(summarize_with_ollama(prompt))
        elif llm == "gemini":
            summaries.append(summarize_with_gemini(prompt))
        elif llm == "gpt":
            summaries.append(summarize_with_gpt(prompt))
        else:
            raise ValueError(f"Unsupported LLM: {llm}")

    print("Chunk summaries:", sum(count_words(s) for s in summaries), "words total.")

    # Combine summaries into a final summary

    print("Combining chunk summaries into final summary...")

    prompt = build_summarization_prompt(summaries, mode="hierarchical")

    if llm == "ollama":
        final_summary = summarize_with_ollama(prompt)
    elif llm == "gemini":
        final_summary = summarize_with_gemini(prompt)
    elif llm == "gpt":
        final_summary = summarize_with_gpt(prompt)
    else:
        raise ValueError(f"Unsupported LLM: {llm}")

    return final_summary

def summarize_progress_direct(progress, llm="ollama", mode="full"):
    """
    Directly summarize the progress without chunking.

    :param progress: dict with progress data
    :param llm: name of the LLM to use for summarization
    :param mode: summarization mode, either "full" or "hierarchical"
    :return: summary text
    """
    text = build_chronological_text_from_progress(progress)
    prompt = build_summarization_prompt(text, mode=mode)

    if llm == "ollama":
        return summarize_with_ollama(prompt)
    elif llm == "gemini":
        return summarize_with_gemini(prompt)
    elif llm == "gpt":
        return summarize_with_gpt(prompt)
    else:
        raise ValueError(f"Unsupported LLM: {llm}")

def summarize_progress_idea_1(progress, llm="ollama", mode="full", check_long_progress=True):
    """
    - If progress length is short enough → single summary
    - If progress length is long → multiple summaries → make final summary from them

    :param progress: dict with progress data
    :param llm: name of the LLM to use for summarization
    :param model: name of the model to use for summarization
    :param mode: summarization mode, either "full" or "hierarchical"
    :param check_long_progress: boolean to check if progress is long enough to split into chunks
    :return: summary text
    """
    if check_long_progress and is_long_progress(progress):
        print("Splitting into chunks for summarization...")
        # Split into chunks if too long
        messages = split_progress_by_length(progress["messages"])

        return summarize_chunked_progress(messages, progress, llm=llm, mode=mode)
    else:
        print("Summarizing directly...")
        # Direct summarization for shorter progress
        text = build_chronological_text_from_progress(progress)
        prompt = build_summarization_prompt(text, mode=mode)
        if llm == "ollama":
            return summarize_with_ollama(prompt)
        elif llm == "gemini":
            return summarize_with_gemini(prompt)
        elif llm == "gpt":
            return summarize_with_gpt(prompt)
        else:
            raise ValueError(f"Unsupported LLM: {llm}")

def summarize_progress_idea_2(progress, llm="ollama", model="gemma3:4b", mode="full"):
    """
    Organize progress threads into groups (annual, semi-annual, etc.) and summarize each group. Then union the summaries while maintaining chronological order.

    :param progress: dict with progress data
    :param llm: name of the LLM to use for summarization
    :param model: name of the model to use for summarization
    :param mode: summarization mode, either "full" or "hierarchical"
    :return: summary text
    :return:
    """

    min_ts = 0
    max_ts = 0

    for msg in progress["messages"]:
        if (min_ts == 0 or msg["timestamp"] < min_ts) and msg["timestamp"] != "0000-00-00 00:00:00":
            min_ts = msg["timestamp"]
        if (max_ts == 0 or msg["timestamp"] > max_ts) and msg["timestamp"] != "0000-00-00 00:00:00":
            max_ts = msg["timestamp"]

    progress_duration_days = (dt.strptime(max_ts, "%Y-%m-%d %H:%M:%S") - dt.strptime(min_ts, "%Y-%m-%d %H:%M:%S")).days

    print(f"Progress duration: {progress_duration_days} days.")

    if progress_duration_days < 180:
        print("Progress is short, summarizing directly...")
        # Direct summarization for short progress
        return summarize_progress_idea_1(progress, llm=llm, mode=mode, check_long_progress=False)

    elif progress_duration_days < 365:
        print("Progress is short-medium, summarizing by quarter...")
        # Split into chunks if too long
        messages = split_progress_by_timeframe(progress, timeframe=1/4)
        return summarize_chunked_progress(messages, progress, llm=llm, mode=mode)

    elif progress_duration_days < 365 * 2:
        print("Progress is medium, summarizing by semester...")
        # Split into chunks if too long
        messages = split_progress_by_timeframe(progress, timeframe=1/2)
        return summarize_chunked_progress(messages, progress, llm=llm, mode=mode)

    else:
        print("Progress is long, summarizing by year...")
        # Split into chunks if too long
        messages = split_progress_by_timeframe(progress, timeframe=1)
        return summarize_chunked_progress(messages, progress, llm=llm, mode=mode)

def validate_summary_with_nli_hybrid(summary, progress, device="cuda", lambda_contradiction=0.5):
    """
    Hybrid SummaC-style scoring with penalties for contradiction and weak support.

    :param summary: Generated summary (string)
    :param progress: Dict with 'messages' list, each containing 'content'
    :param device: 'cuda' or 'cpu'
    :param lambda_contradiction: weight for contradiction penalty
    :return: float, adjusted factual consistency score
    """
    model_name = "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()

    id2label = model.config.id2label
    label2id = model.config.label2id
    entail_id = label2id["entailment"]
    contradict_id = label2id["contradiction"]

    # Prepare text
    doc_text = strip_markdown_and_html(" ".join(msg["content"] for msg in progress["messages"]))
    summary_text = strip_markdown_and_html(strip_summary_metadata(summary))

    doc_sentences = clean_sentences(sent_tokenize(doc_text, language='italian'))
    summary_sentences = clean_sentences(sent_tokenize(summary_text, language='italian'))
    if not doc_sentences or not summary_sentences:
        return 0.0

    # Generate all (doc, summary) pairs
    pairs = [{"premise": d, "hypothesis": s} for s in summary_sentences for d in doc_sentences]
    dataset = Dataset.from_list(pairs)

    def predict_batch(batch):
        inputs = tokenizer(batch["premise"], batch["hypothesis"], padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1)

        entail_scores = probs[:, entail_id].tolist()
        contradict_scores = probs[:, contradict_id].tolist()

        return {
            "entailment_score": entail_scores,
            "contradiction_score": contradict_scores,
            "hypothesis": batch["hypothesis"]
        }

    result_dataset = dataset.map(predict_batch, batched=True, batch_size=64)
    df = result_dataset.to_pandas()

    # Group by each summary sentence (hypothesis)
    grouped = df.groupby("hypothesis")
    max_entail = grouped["entailment_score"].max()
    max_contradict = grouped["contradiction_score"].max()

    # Compute base score
    base_score = max_entail.mean()

    # Penalty for contradiction
    contradiction_penalty = lambda_contradiction * max_contradict.mean()

    # Final hybrid score
    adjusted_score = base_score - contradiction_penalty
    return round(max(base_score, 0.0), 4), round(max(adjusted_score, 0.0), 4)

def validate_summary_with_qags_batched(summary, progress, device=0):
    """
    Compute QAGS score using batched question generation and QA.
    """
    # Load models
    qg = pipeline("text2text-generation", model="it5/it5-large-question-generation", device=device)
    qa = pipeline("question-answering", model="osiria/bert-italian-cased-question-answering", device=device)

    # Build document and split summary
    document = strip_markdown_and_html(" ".join(msg["content"] for msg in progress["messages"]))

    summary_clean = strip_markdown_and_html(strip_summary_metadata(summary))
    sentences = sent_tokenize(summary_clean, language='italian')
    sentences = clean_sentences(sentences)

    # Only now call QG
    questions = qg(sentences, batch_size=8)
    # 1. Generate questions in batch
    questions = [q["generated_text"] for q in questions]
    questions = clean_sentences(questions)
    questions = [q for q in questions if not is_question_garbage(q)]
    questions = remove_near_duplicates(questions)

    print(f"Generated {len(questions)} questions: {questions}")

    # 2. Build Dataset for QA calls
    qa_data = Dataset.from_list([
        {"question": q, "context_doc": document, "context_sum": summary}
        for q, summary in zip(questions, [summary] * len(questions))
    ])

    # 3. Run QA on document and summary
    def qa_batch_context_doc(batch):
        context = [batch["context_doc"][0]] * len(batch["question"])
        answers = qa(question=batch["question"], context=context)
        if isinstance(answers, dict):
            answers = [answers]
        return {"answer_doc": [a["answer"] for a in answers]}

    def qa_batch_context_sum(batch):
        context = [batch["context_sum"][0]] * len(batch["question"])
        answers = qa(question=batch["question"], context=context)
        if isinstance(answers, dict):
            answers = [answers]
        return {"answer_sum": [a["answer"] for a in answers]}

    qa_data = qa_data.map(qa_batch_context_doc, batched=True, batch_size=8)
    qa_data = qa_data.map(qa_batch_context_sum, batched=True, batch_size=8)

    # 4. Compare answers
    df = qa_data.to_pandas()
    # print("Answers from document:", df["answer_doc"].tolist())
    # print("Answers from summary:", df["answer_sum"].tolist())
    match_count = sum(
        1 for a, b in zip(df["answer_doc"], df["answer_sum"])
        if a and b and a.strip().lower() == b.strip().lower()
    )

    match_count = sum(1 for a, b in zip(df["answer_doc"], df["answer_sum"]) if a and b and is_similar(a, b))

    score = round(match_count / len(df), 4) if len(df) else 0.0
    print(f"QAGS score: {score} ({match_count} matches out of {len(df)}) for progress ID {progress['progress_id']}")
    return score

def gpt_score_summary(summary, progress, model="gpt-4.1-mini"):
    system_msg = (
        "Sei un revisore esperto di contenuti. "
        "Il tuo compito è valutare la fedeltà fattuale di un riassunto rispetto al documento originale. "
        "Oltre alla fedeltà, considera anche la lunghezza del riassunto, tenendo conto che un riassunto troppo lungo potrebbe essere meno utile. "
        "Un compression ratio ideale è tra 0.15 e 0.5 rispetto al testo originale, ma non deve superare il 60% della lunghezza del testo originale altrimenti il punteggio sarà 1."
        "Se il riassunto è scritto in una lingua diversa dall'italiano, il punteggio sarà 1."
        "Assegna un punteggio da 1 a 5 dove:\n"
        "1 = Molto impreciso, con gravi errori o allucinazioni\n"
        "2 = Alcune affermazioni errate o non supportate\n"
        "3 = Parzialmente fedele, ma con omissioni o vaghezze\n"
        "4 = Generalmente fedele con minime inesattezze\n"
        "5 = Completamente fedele e preciso\n\n"
    )

    full_text = strip_markdown_and_html(" ".join(msg["content"] for msg in progress["messages"]))

    user_msg = (
        f"Testo originale:\n{full_text}\n\n"
        f"Riassunto:\n{strip_markdown_and_html(summary)}\n\n"
        f"Valuta con un formato del tipo:\nPunteggio: <numero>\nSpiegazione: <testo>"
    )

    client = OpenAI(api_key=api_keys.OPENAI_API_KEY)

    response = None

    while response is None:
        try:
            response = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ]
            )
        except Exception as e:
            print(f"Error generating content with GPT: {e}, sleeping for 10 seconds before retrying...")
            sleep(10)
            response = None

    content = response.output_text.strip()

    # print("GPT response content:", content)

    match = re.search(r"Punteggio\s*[:=]?\s*(\d)", content)
    score = int(match.group(1)) if match else None

    print(f"GPT score: {score} for progress ID {progress['progress_id']}")

    return score