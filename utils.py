# Create summarization utilities module
import textwrap
import ollama
from markdown_it import MarkdownIt
from bs4 import BeautifulSoup
import re

def is_long_progress(progress, msg_threshold=30, char_threshold=15000):
    return progress["message_count"] >= msg_threshold or progress["total_char_length"] >= char_threshold

def count_words(text):
    words = re.findall(r'\b\w+\b', text, re.UNICODE)
    return len(words)

def build_chronological_text_from_progress(progress):
    # Build a chronological text from messages, tokenizing by author and content

    messages = progress.get("messages", [])

    if not messages:
        return ""

    # Sort messages by message_id to ensure chronological order
    messages = sorted(messages, key=lambda x: x['message_id'])

    # Join messages into a single string with author and content
    # Use '\\n\\n' to separate messages for better readability

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

def split_into_chunks(messages, char_threshold=10000):
    chunks = []
    current_chunk = []
    current_length = 0
    for msg in messages:
        msg_length = len(msg.get("content", ""))
        if current_length + msg_length > char_threshold and current_chunk:
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

    max_length = max(200, min(1000, count_words(text_for_prompt) // 4))

    print(f"Max length for summarization: {max_length} words.")

    if mode == "full":
        prompt = textwrap.dedent(f"""
        Istruzioni:
            Riassumi il seguente dettaglio di progetto in formato cronologico
            Usa un formato ad elenco puntato per gli eventi principali.
            Sottolinea le persone coinvolte e le loro azioni.
            Sottolinea la presenza di allegati ai messaggi.
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

def summarize_with_ollama(prompt, model):
    response = ollama.generate(
        model=model,
        prompt=prompt,
    )

    print("Summary length:", count_words(response["response"]), "words.")

    return response["response"]

def summarize_progress_idea_1(progress, llm="ollama", model="gemma3:4b", mode="full"):

    if is_long_progress(progress):
        print("Progress is long, splitting into chunks for summarization...")
        # Split into chunks if too long
        progress_for_summary = progress
        messages = split_into_chunks(progress["messages"])
        summaries = []
        for chunk in messages:
            progress_for_summary["messages"] = chunk
            text = build_chronological_text_from_progress(progress_for_summary)
            prompt = build_summarization_prompt(text, mode=mode)
            if llm == "ollama":
                summaries.append(summarize_with_ollama(prompt, model=model))
            elif llm == "gemini":
                # PLH
                raise NotImplementedError("Gemini summarization not implemented yet.")
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
            # PLH
            raise NotImplementedError("Gemini summarization not implemented yet.")
        elif llm == "gpt":
            # PLH
            raise NotImplementedError("GPT summarization not implemented yet.")
        else:
            raise ValueError(f"Unsupported LLM: {llm}")

        return final_summary
    else:
        print("Progress is short, summarizing directly...")
        # Direct summarization for shorter progress
        text = build_chronological_text_from_progress(progress)
        prompt = build_summarization_prompt(text, mode=mode)
        if llm == "ollama":
            return summarize_with_ollama(prompt, model=model)
        elif llm == "gemini":
            # PLH
            raise NotImplementedError("Gemini summarization not implemented yet.")
        elif llm == "gpt":
            # PLH
            raise NotImplementedError("GPT summarization not implemented yet.")
        else:
            raise ValueError(f"Unsupported LLM: {llm}")