from youtube_transcript_api import YouTubeTranscriptApi

# 1. Video ID
video_id = "a_odruU6jvg"

# 2. Instantiate the API and fetch transcript
ytt_api = YouTubeTranscriptApi()
fetched_transcript = ytt_api.fetch(video_id)

# 3. Open output file
output_file = r"C:\b-secur\local_data\dgr\transcriptstranscript.txt"
with open(output_file, "w", encoding="utf-8") as f:
    for snippet in fetched_transcript.snippets:
        start = snippet.start         # start time in seconds
        duration = snippet.duration   # duration in seconds
        text = snippet.text           # transcript text
        
        # Format as [start --> end] text
        f.write(f"[{start:.2f} --> {start + duration:.2f}] {text}\n")

print(f"Transcript saved to: {output_file}")
