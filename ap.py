

from pytube import YouTube

# Get the YouTube video
yt = YouTube("https://www.youtube.com/watch?v=H598lrc7SIw")

# Attempt to get the transcript in Hindi
try:
    transcript = yt.captions.get_by_language_code('en')  # Change 'hi' to 'en' for English captions
    if transcript:
        # Generate SRT captions
        transcript_text = transcript.generate_srt_captions()
        
        # Print the transcript text
        print(transcript_text)
    else:
        print("No transcript available in the specified language.")
except Exception as e:
    print(f"An error occurred: {e}")
