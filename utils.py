# Create summarization utilities module
import textwrap
import ollama
from transformers import AutoTokenizer, AutoModelForCausalLM

def is_long_progress(progress, msg_threshold=300, char_threshold=200000):
    return progress["message_count"] >= msg_threshold or progress["total_char_length"] >= char_threshold

def build_chronological_text(messages):
    return "\n\n".join(f"{m['author']} - {m['timestamp']}:\n{m['content']}" for m in messages).strip()

def build_chronological_text_from_progress(progress):
    # Build a chronological text from messages, tokenizing by author and content

    messages = progress.get("messages", [])

    if not messages:
        return ""

    # Sort messages by message_id to ensure chronological order
    messages = sorted(messages, key=lambda x: x['message_id'])

    # Join messages into a single string with author and content
    # Use '\\n\\n' to separate messages for better readability

    text  = f"{progress['subject']} - {progress['created_at']}\n"
    text += f"{progress['description']}\n\n"
    text += "Messages:\n\n"

    for m in messages:
        m["content"] = m.get("content", "")
        text += f"{m['author']} - {m['timestamp']}: {m['content']}\n\n"

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

def build_summarization_prompt(text, mode="full"):
    if mode == "full":
        prompt = textwrap.dedent(f"""\
        Sei un esperto di sintesi di testi e devi riassumere in modo dettagliato il seguente dettaglio di un progetto, mantenendo l'ordine cronologico degli eventi e le informazioni chiave.
        Per favore, crea un riassunto coerente e conciso che evidenzi i punti principali e le azioni intraprese.
        Usa un linguaggio chiaro e preciso, evitando ambiguità o informazioni superflue.
        Non essere prolisso, ma assicurati di includere tutti gli aspetti rilevanti, come le decisioni prese, le azioni svolte e i risultati ottenuti.
        È fondamentale mantenere il testo originale in italiano.
        Per favore, non aggiungere introduzioni o conclusioni al riassunto generato.
        
        {text if isinstance(text, str) else '\\n\\n'.join(text)}
        """)
    elif mode == "hierarchical":
        prompt = textwrap.dedent(f"""\
        Sei un esperto di sintesi di testi. Hai già sintetizzato i dettagli di un progetto in più parti, e ora devi creare un riassunto finale che integri tutte le parti precedenti.
        Di seguito trovi i riassunti delle parti precedenti, che devono essere combinati in un unico riassunto coerente e conciso.
        Per favore, crea un riassunto coerente e conciso che evidenzi i punti principali e le azioni intraprese.
        Usa un linguaggio chiaro e preciso, evitando ambiguità o informazioni superflue.
        Non essere prolisso, ma assicurati di includere tutti gli aspetti rilevanti, come le decisioni prese, le azioni svolte e i risultati ottenuti.
        È fondamentale mantenere il testo originale in italiano.
        Per favore, non aggiungere introduzioni o conclusioni al riassunto generato.
        
        {text if isinstance(text, str) else '\\n\\n'.join(text)}
        """)
    else:
        raise ValueError("Unsupported prompt mode.")

    # print(prompt)
    return prompt.strip()

def summarize_with_ollama(prompt, model):
    response = ollama.generate(
        model=model,
        prompt=prompt,
    )
    return response["response"]

def summarize_progress(progress, llm="ollama", model="gemma3:12b", mode="full"):

    if is_long_progress(progress):
        print("Progress is long, splitting into chunks for summarization...")
        # Split into chunks if too long
        messages = split_into_chunks(progress["messages"])
        summaries = []
        for chunk in messages:
            text = build_chronological_text(chunk)
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

        print("Chunk summaries:", sum(len(s) for s in summaries), "characters total.")

        # Combine summaries into a final summary

        print("Combining chunk summaries into final summary...")

        if llm == "ollama":
            prompt = build_summarization_prompt(summaries, mode="hierarchical")
            final_summary = summarize_with_ollama(prompt, model=model)
        elif llm == "gemini":
            # PLH
            raise NotImplementedError("Gemini summarization not implemented yet.")
        elif llm == "gpt":
            # PLH
            raise NotImplementedError("GPT summarization not implemented yet.")
        else:
            raise ValueError(f"Unsupported LLM: {llm}")

        print("Final summary length:", len(final_summary), "characters.")

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