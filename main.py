from generate_transcript import transcribe_audio
from summarize_text import summarize_text_bart, summarize_text_t5
from translate import translate_file
import torch
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

def main():
    if torch.cuda.is_available():
        print("CUDA is available. Number of GPUs:", torch.cuda.device_count())
        print("GPU Name:", torch.cuda.get_device_name(0))
        print()
    else:
        print("CUDA is not available.\n")

    print(
        "Welcome to the Audio Processing Utility. This tool performs several functions on audio files, primarily focused on processing spoken English content.\n"
        "Here’s what the utility will do:\n\n"
        "- **Transcription**: Converts spoken language within an MP3 file into text, creating a detailed transcript that includes timestamps for each spoken segment.\n"
        "                     This is useful for generating accurate records of speeches, meetings, or any audio content.\n\n"
        "- ** Translation **: Translates the English transcript into Chinese, maintaining the original timeline structure.\n"
        "                     This feature is particularly useful for multilingual audiences and ensures that the translated content aligns with the timing of the spoken words in the audio.\n\n"
        "- **Summarization**: Utilizes two different models, Facebook's BART-large-CNN and FalconSai's text summarization model, to condense the content of the transcript into a shorter form.\n"
        "                     Providing both timestamped and plain text summaries, this helps in quickly understanding the key points without needing to go through the entire transcript.\n"
        "                     The output will include files with detailed timelines as well as simplified text versions, making it suitable for various use cases such as subtitling, reviews, or content analysis.\n"
    )

    audio_path = input("Enter the path to the audio file: ")
    times, transcript_texts = transcribe_audio(audio_path)

    transcript_base_name = os.path.splitext(audio_path)[0]
    timeline_filename = f"{transcript_base_name}_transcript_with_timeline.txt"
    notimeline_filename = f"{transcript_base_name}_transcript_without_timeline.txt"

    translation_output_path = f"{transcript_base_name}_translated_to_chinese_with_timeline.txt"

    # Translate the transcript with timeline
    translate_file(timeline_filename, translation_output_path)

    # Attempt to read the transcript file with Big5 encoding, then UTF-8 if Big5 fails
    try:
        with open(notimeline_filename, "r", encoding='big5') as file:
            transcript_text = file.read()
    except UnicodeDecodeError:
        print("Failed to read the file with Big5 encoding. Trying with utf-8.")
        try:
            with open(notimeline_filename, "r", encoding='utf-8') as file:
                transcript_text = file.read()
        except UnicodeDecodeError:
            print("Failed to read the file with utf-8 encoding. Please check the file encoding.")
            return

    summaries_with_times_bart = summarize_text_bart(transcript_texts, times, 150)
    summaries_with_times_t5 = summarize_text_t5(transcript_texts, times, 150)

    # Paths for BART summaries
    summary_path_without_timeline_bart = f"{transcript_base_name}_summary_without_timeline_bart.txt"
    summary_path_with_timeline_bart = f"{transcript_base_name}_summary_with_timeline_bart.txt"

    # Paths for T5 summaries
    summary_path_without_timeline_t5 = f"{transcript_base_name}_summary_without_timeline_t5.txt"
    summary_path_with_timeline_t5 = f"{transcript_base_name}_summary_with_timeline_t5.txt"

    # Save BART summaries
    with open(summary_path_without_timeline_bart, "w", encoding='utf-8') as file:
        file.write(' '.join([summary for summary, _ in summaries_with_times_bart]))
    with open(summary_path_with_timeline_bart, "w", encoding='utf-8') as file:
        for summary, (start, end) in summaries_with_times_bart:
            file.write(f"[{start:.2f}s -> {end:.2f}s] {summary}\n\n")

    # Save T5 summaries
    with open(summary_path_without_timeline_t5, "w", encoding='utf-8') as file:
        file.write(' '.join([summary for summary, _ in summaries_with_times_t5]))
    with open(summary_path_with_timeline_t5, "w", encoding='utf-8') as file:
        for summary, (start, end) in summaries_with_times_t5:
            file.write(f"[{start:.2f}s -> {end:.2f}s] {summary}\n\n")

    print()
    print(
        f"Summary files saved:\n- {summary_path_without_timeline_bart}\n- {summary_path_with_timeline_bart}\n- {summary_path_without_timeline_t5}\n- {summary_path_with_timeline_t5}")

if __name__ == "__main__":
    main()

