from faster_whisper import WhisperModel
import os

def transcribe_audio(audio_path):
    model_size = "large-v2"
    model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")

    # Transcribe the audio
    segments, info = model.transcribe(audio_path, beam_size=5)
    print("\nDetected language '%s' with probability %f\n" % (info.language, info.language_probability))

    with_timeline = []
    without_timeline = []
    times = []

    # Extract base name from the audio path and prepare output file names
    transcript_base_name = os.path.splitext(audio_path)[0]
    timeline_filename = f"{transcript_base_name}_transcript_with_timeline.txt"
    notimeline_filename = f"{transcript_base_name}_transcript_without_timeline.txt"

    # Compile lines and corresponding times
    for segment in segments:
        line_with_time = "[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text)
        with_timeline.append(line_with_time)
        without_timeline.append(segment.text)
        times.append((segment.start, segment.end))
        print(f"Line: {line_with_time}")  # Print each line with its timeline

    # Save transcripts with timeline
    with open(timeline_filename, "w") as f:
        f.write("\n".join(with_timeline))
    print()
    print(f"Transcript with timeline saved to: {timeline_filename}\n")

    # Save transcripts without timeline
    with open(notimeline_filename, "w") as f:
        f.write("\n".join(without_timeline))
    print(f"Transcript without timeline saved to: {notimeline_filename}\n")

    print("Transcription completed. Check the output files for details.\n")

    return times, without_timeline  # Ensure times and texts are returned for further processing

