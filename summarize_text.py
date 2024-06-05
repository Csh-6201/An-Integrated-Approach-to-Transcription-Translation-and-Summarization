from transformers import pipeline, BartTokenizer

def summarize_text_bart(transcript_texts, times, max_seconds=150):

    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    summaries_with_times = []
    current_segment = ""
    current_start_time = times[0][0]
    segment_end_time = current_start_time
    print("Generate summary by using facebook/bart-large-cnn...\n")

    for (start, end), text in zip(times, transcript_texts):
        if (end - current_start_time) > max_seconds:
            if current_segment:
                inputs = tokenizer(current_segment, return_tensors="pt", truncation=True, max_length=1024)
                summary_ids = summarizer.model.generate(**inputs, max_length=200, min_length=40, do_sample=False)
                summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                summaries_with_times.append((summary, (current_start_time, segment_end_time)))
                print(f"[{current_start_time:.2f}s -> {segment_end_time:.2f}s] {summary}\n")  # Print each summary with its timeline
            current_segment = text
            current_start_time = start
        else:
            current_segment += " " + text
        segment_end_time = end

    if current_segment:
        inputs = tokenizer(current_segment, return_tensors="pt", truncation=True, max_length=1024)
        summary_ids = summarizer.model.generate(**inputs, max_length=200, min_length=40, do_sample=False)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries_with_times.append((summary, (current_start_time, segment_end_time)))
        print(f"[{current_start_time:.2f}s -> {segment_end_time:.2f}s] {summary}")  # Print the final summary with its timeline

    return summaries_with_times  # Make sure to return the result

def summarize_text_t5(transcript_texts, times, max_seconds=150):
    summarizer = pipeline("summarization", model="Falconsai/text_summarization")

    summaries_with_times = []
    current_segment = ""
    current_start_time = times[0][0]
    segment_end_time = current_start_time
    print()
    print("Generate summary by using Falconsai/text_summarization...\n")

    for (start, end), text in zip(times, transcript_texts):
        if (end - current_start_time) > max_seconds:
            if current_segment:
                summary = summarizer(current_segment, max_length=200, min_length=40, do_sample=False)[0]['summary_text']
                summaries_with_times.append((summary, (current_start_time, segment_end_time)))
                print(f"[{current_start_time:.2f}s -> {segment_end_time:.2f}s] {summary}\n")  # Print each summary with its timeline
            current_segment = text
            current_start_time = start
        else:
            current_segment += " " + text
        segment_end_time = end

    # Handle the last segment if there is any text left
    if current_segment:
        summary = summarizer(current_segment, max_length=200, min_length=40, do_sample=False)[0]['summary_text']
        summaries_with_times.append((summary, (current_start_time, segment_end_time)))
        print(f"[{current_start_time:.2f}s -> {segment_end_time:.2f}s] {summary}")  # Print the final summary with its timeline

    return summaries_with_times



