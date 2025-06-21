# Create summarization utilities module
import textwrap
import requests
import ollama

def is_long_progress(progress, msg_threshold=50, char_threshold=100000):
    return progress["message_count"] >= msg_threshold or progress["total_char_length"] >= char_threshold

def build_chronological_text(messages):
    return "\\n\\n".join([f"{m['author']}: {m['content'].strip()}" for m in sorted(messages, key=lambda x: x['message_id'])])

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
        Riassumi in modo dettagliato la seguente conversazione, mantenendo l'ordine cronologico degli eventi.
        Non aggiungere introduzioni o conclusioni al riassunto.
        
        {text}
        """)
    elif mode == "hierarchical":
        prompt = textwrap.dedent(f"""\
        Riassumi in modo dettagliato i seguenti riassunti parziali in una sintesi coerente che rispetti l'ordine temporale e sottolinei i passaggi fondamentali.
        Non aggiungere introduzioni o conclusioni al riassunto.
        
        {text}
        """)
    else:
        raise ValueError("Unsupported prompt mode.")

    # print(prompt)
    return prompt.strip()

def summarize_with_ollama(prompt, model):
    response = ollama.generate(
        model=model,
        prompt=prompt,
        stream=False
    )
    return response["response"]

def summarize_progress(progress, llm="ollama", model="gemma3:4b", mode="full"):

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
        messages = progress["messages"]
        text = build_chronological_text(messages)
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