# Create summarization utilities module
import textwrap
from collections import defaultdict
from datetime import datetime as dt

import ollama
from google import genai
from bs4 import BeautifulSoup
from markdown_it import MarkdownIt
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize

import api_keys

def is_long_progress(progress, msg_threshold=30, char_threshold=15000):
    return progress["message_count"] >= msg_threshold or progress["total_char_length"] >= char_threshold

def count_words(text):
    tokens = word_tokenize(text, language='italian')
    return len([t for t in tokens if t.isalpha()])

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

def split_progress_by_length(messages, words_threshold=500):
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

    max_length = max(200, min(1000, count_words(text_for_prompt) // 2))

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
            Lunghezza massima: {max_length} parole.

        Dettaglio del progetto:
        {text_for_prompt}
        """)
    elif mode == "hierarchical":
        prompt = textwrap.dedent(f"""
        Istruzioni:
            Unisci i testi forniti in un unico riassunto coerente e dettagliato.
        Vincoli:
            Scrivi in italiano.
            Non aggiungere introduzioni né conclusioni.
            Lunghezza massima: {max_length} parole.
            Evita ripetizioni.
            Il testo deve essere fluido, logico e leggibile.
            
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

def summarize_with_ollama(prompt, model):
    response = ollama.generate(
        model=model,
        prompt=prompt,
    )

    print("Summary length:", count_words(response["response"]), "words.")

    return response["response"]

def summarize_with_gemini(prompt, model):
    google_client = genai.Client(api_key=api_keys.GOOGLE_API_KEY)
    response = google_client.models.generate_content(
        model=model,
        contents=prompt,
    )
    print("Summary length:", count_words(response.text), "words.")
    return response.text

def summarize_chunked_progress(messages, progress_for_summary, llm, model, mode):
    summaries = []
    for chunk in messages:
        progress_for_summary["messages"] = chunk
        text = build_chronological_text_from_progress(progress_for_summary)
        prompt = build_summarization_prompt(text, mode=mode)
        if llm == "ollama":
            summaries.append(summarize_with_ollama(prompt, model=model))
        elif llm == "gemini":
            summaries.append(summarize_with_gemini(prompt, model=model))
        elif llm == "gpt":
            # PLH
            raise NotImplementedError("GPT summarization not implemented yet.")
        else:
            raise ValueError(f"Unsupported LLM: {llm}")

    print("Chunk summaries:", sum(count_words(s) for s in summaries), "words total.")

    # Combine summaries into a final summary

    print("Combining chunk summaries into final summary...")

    prompt = build_summarization_prompt(summaries, mode="hierarchical")

    if llm == "ollama":
        final_summary = summarize_with_ollama(prompt, model=model)
    elif llm == "gemini":
        final_summary = summarize_with_gemini(prompt, model=model)
    elif llm == "gpt":
        # PLH
        raise NotImplementedError("GPT summarization not implemented yet.")
    else:
        raise ValueError(f"Unsupported LLM: {llm}")

    return final_summary

def summarize_progress_idea_1(progress, llm="ollama", model="gemma3:4b", mode="full", check_long_progress=True):
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

        return summarize_chunked_progress(messages, progress, llm=llm, model=model, mode=mode)
    else:
        print("Summarizing directly...")
        # Direct summarization for shorter progress
        text = build_chronological_text_from_progress(progress)
        prompt = build_summarization_prompt(text, mode=mode)
        if llm == "ollama":
            return summarize_with_ollama(prompt, model=model)
        elif llm == "gemini":
            return summarize_with_gemini(prompt, model=model)
        elif llm == "gpt":
            # PLH
            raise NotImplementedError("GPT summarization not implemented yet.")
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
        return summarize_progress_idea_1(progress, llm=llm, model=model, mode=mode, check_long_progress=False)

    elif progress_duration_days < 365:
        print("Progress is short-medium, summarizing by quarter...")
        # Split into chunks if too long
        messages = split_progress_by_timeframe(progress, timeframe=1/4)
        return summarize_chunked_progress(messages, progress, llm=llm, model=model, mode=mode)

    elif progress_duration_days < 365 * 2:
        print("Progress is medium, summarizing by semester...")
        # Split into chunks if too long
        messages = split_progress_by_timeframe(progress, timeframe=1/2)
        return summarize_chunked_progress(messages, progress, llm=llm, model=model, mode=mode)

    else:
        print("Progress is long, summarizing by year...")
        # Split into chunks if too long
        messages = split_progress_by_timeframe(progress, timeframe=1)
        return summarize_chunked_progress(messages, progress, llm=llm, model=model, mode=mode)