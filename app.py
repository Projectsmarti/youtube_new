# import streamlit as st
# import re, os, ast, speech_recognition as sr
# import google.generativeai as genai
# from datetime import datetime, timedelta
# from youtube_transcript_api import YouTubeTranscriptApi
# from pytube import YouTube
# import time
# from moviepy.editor import AudioFileClip

# prompt="""You are Youtube video summarizer. You will be taking the transcript text and summarizing the entire video and providing
#                     the overall summary get the major points discussed,never mention the name odf the person
#                     within 250 words in following format. Please provide the summary of the text given here:"""

# def get_video_duration(youtube_video_url):
#     try:
#         # Fetch video details using pytube
#         yt = YouTube(youtube_video_url)
#         duration = yt.length  # Duration in seconds

#         # Convert duration to minutes and seconds
#         minutes, seconds = divmod(duration, 60)
#         duration_formatted = f"{minutes} minutes and {seconds} seconds"

#         return duration_formatted

#     except Exception as e:
#         raise e

# # Function to extract transcript details
# def extract_transcript_details(youtube_video_url):
#     try:
#         # Extract video ID from the URL
#         if "youtu.be" in youtube_video_url:
#             video_id = youtube_video_url.split("/")[-1]
#         else:
#             video_id = youtube_video_url.split("=")[1]

#         # Fetch the video transcript (you can specify the language)
#         transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=("en",))  # English transcript

#         # Initialize a dictionary to store timestamps and transcriptions
#         transcript_dict = {}

#         # Extract text content and clean data
#         for t in transcript:
#             cleaned_text = re.sub(r"[^a-zA-Z0-9-ışğöüçiIŞĞÖÜÇİ ]", "", t["text"])
#             if cleaned_text:
#                 timestamp = t["start"]
#                 formatted_time = f"{int(timestamp // 3600):02d}:{int((timestamp // 60) % 60):02d}:{int(timestamp % 60):02d}"
#                 transcript_dict[formatted_time] = cleaned_text

#         return transcript_dict

#     except Exception as e:
#         st.error(f"Error: {e}")
#         return None
      
# # Function to merge transcript
# def merge_transcript(transcript, interval_minutes=5):
#     try:
#         # Initialize variables
#         interval = timedelta(minutes=interval_minutes)
#         merged_transcript = {}
#         current_interval_start = datetime.strptime('00:00:00', '%H:%M:%S')
#         current_interval_end = current_interval_start + interval
#         current_text = ""

#         for timestamp, text in transcript.items():
#             time = datetime.strptime(timestamp, '%H:%M:%S')

#             # If the current timestamp is within the current interval, append the text
#             if time < current_interval_end:
#                 current_text += text + " "
#             else:
#                 # Save the current interval text to the merged_transcript
#                 merged_transcript[current_interval_end.strftime('%H:%M:%S')] = current_text.strip()
#                 # Move to the next interval
#                 while time >= current_interval_end:
#                     current_interval_start = current_interval_end
#                     current_interval_end = current_interval_start + interval
#                 # Start the new interval with the current text
#                 current_text = text + " "

#         # Add the last interval if there's remaining text
#         if current_text:
#             merged_transcript[current_interval_end.strftime('%H:%M:%S')] = current_text.strip()

#         # Ensure the last key matches the last key from the input transcript
#         last_key_input = list(transcript.keys())[-1]
#         last_key_output = list(merged_transcript.keys())[-1]
#         if last_key_input != last_key_output:
#             merged_transcript[last_key_input] = merged_transcript.pop(last_key_output)

#         return merged_transcript

#     except Exception as e:
#         st.error(f"Error: {e}")
#         return None

# def generate_gemini_content(transcript_text, prompt, max_output_tokens=None):
#   with st.spinner('Summarizing...'):
#     try:
#         os.environ['GOOGLE_API_KEY'] = 'AIzaSyBbepUh8x3CqpkxNFnJ1IX0dFc0UNTwwbU'  # Replace with your API key
#         genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

#         models = [m for m in genai.list_models() if "text-bison" in m.name]
#         model = models[0].name

#         if max_output_tokens is None:
#             max_output_tokens = 300

#         # Generate text using the specified model
#         completion = genai.generate_text(
#             model=model,
#             prompt=transcript_text + prompt,
#             temperature=0.1,
#             max_output_tokens=max_output_tokens
#         )

#         output = completion.result.split('\n')
#         response = "\n".join(output)
#         return response
#     except Exception as e:
#         st.error("Can't generate summary")
#         return ""
      
# # def delete_file(file_path):
# #     try:
# #         os.remove(file_path)
# #         print(f"Deleted file: {file_path}")
# #     except Exception as e:
# #         print(f"An error occurred while deleting the file: {e}")

# output_path = '/tmp/amios'
# if not os.path.exists(output_path):
#     os.makedirs(output_path)
  
# def convert_to_wav(input_file, output_file):
#   with st.spinner('Converting audio to WAV...'):
#     try:
#         audio_clip = AudioFileClip(input_file)
#         audio_clip.write_audiofile(output_file)
#         print(f"Converted {input_file} to {output_file}")
#     except Exception as e:
#         print(f"An error occurred while converting to WAV: {e}")

# def download_audio(video_url, output_path):
#   with st.spinner('Downloading audio...'):
#     try:
#         yt = YouTube(video_url)
#         video_title = yt.title
#         audio_stream = yt.streams.filter(only_audio=True).first()
#         downloaded_file = audio_stream.download(output_path=output_path)
#         st.write(f"Audio downloaded successfully to {downloaded_file}")
#         return downloaded_file, video_title
#     except Exception as e:
#         st.error(f"An error occurred: {e}")
#         return None, None

# def transcribe_audio(file_path, language='en-US', retries=3):
#   with st.spinner('Transcribing audio...'):
#     r = sr.Recognizer()
#     for attempt in range(retries):
#         try:
#             with sr.AudioFile(file_path) as source:
#                 audio_text = r.record(source)
#                 text = r.recognize_google(audio_text, language=language)
#                 st.write("Transcription completed successfully")
#                 return text
#         except sr.UnknownValueError:
#             st.write("Could not understand audio")
#             return ""
#         except sr.RequestError as req:
#             st.write(f"Error fetching results from Google Web Speech API: {req}")
#             if attempt < retries - 1:
#                 time.sleep(2)  # Wait for 2 seconds before retrying
#             else:
#                 return f"Transcription failed after {retries} attempts."
              
# #summarizer
# def main1():
#     st.markdown(
#             """
#             <style>
#             .stButton > button {
#                 display: block;
#                 margin: 0 auto;
#             }
#             </style>
#             """, unsafe_allow_html=True)
#     # Streamlit UI
#     st.title("YouTube Video Transcript Summarizer")
#     # Initialize session state variables
#     if 'youtube_link' not in st.session_state:
#         st.session_state.youtube_link = ""
#     if 'interval_minutes' not in st.session_state:
#         st.session_state.interval_minutes = 5
#     if 'transcript_extracted' not in st.session_state:
#         st.session_state.transcript_extracted = False
#     if 'merged_text' not in st.session_state:
#         st.session_state.merged_text = ""
#     if 'show_transcript' not in st.session_state:
#         st.session_state.show_transcript = False

#     youtube_link = st.text_input("Enter YouTube Video Link:")

#     if youtube_link:
#         st.session_state.youtube_link = youtube_link

#     col1, col2 = st.columns([5, 3])

#     if st.session_state.youtube_link:
#         with col1:
#             video_id = st.session_state.youtube_link.split("=")[1]
#             st.video(f"https://www.youtube.com/watch?v={video_id}", format="video/mp4")
#         dur = get_video_duration(st.session_state.youtube_link)
#         st.info(dur)
#         # interval_minutes = st.slider("Select interval in minutes", min_value=1, max_value=10, value=5, step=1)
#         # if interval_minutes:
#         #     st.session_state.interval_minutes = interval_minutes

#         if st.button('Get Summary'):
#             with st.spinner("Extracting transcript..."):
#                 try:
#                     transcript_text = extract_transcript_details(st.session_state.youtube_link)
#                     if transcript_text:
#                         st.success("Transcript extracted successfully!")
#                         try:
#                             # merged_transcript = merge_transcript(transcript_text, st.session_state.interval_minutes)
#                             merged_transcript = merge_transcript(transcript_text, 4)
#                             if merged_transcript:
#                                 merged_text = ""
#                                 for timestamp, text in merged_transcript.items():
#                                     merged_text += f"{timestamp}: {text}\n"
#                                 st.session_state.merged_text = merged_text
#                                 st.session_state.transcript_extracted = True
#                                 if st.session_state.transcript_extracted:
#                                     with st.expander('Show Transcript'):
#                                         st.session_state.show_transcript = True
#                                         if st.session_state.show_transcript:
#                                             st.write(st.session_state.merged_text)
#                                 try:
#                                     res = generate_gemini_content(str(merged_text),prompt)
#                                     st.markdown('## **Summary**')
#                                     st.write(res)
#                                 except:
#                                     st.exception('Failed to get summary')
#                         except Exception as e:
#                             st.exception(f"Failed to merge transcript: {e}")
#                 except Exception as e:
#                     st.exception(f"Failed to extract transcript: {e}")

# #keyword extractor
# def main2():
#     st.title("YouTube Video Keyword Content Analyzer")

#     # Initialize session state variables
#     if 'selected_keywords' not in st.session_state:
#         st.session_state.selected_keywords = []
#     if 'gemini_response' not in st.session_state:
#         st.session_state.gemini_response = ""

#     # Input for YouTube video link
#     youtube_link = st.text_input("Enter YouTube Video Link:")
#     col1, col2 = st.columns([5, 3])

#     if youtube_link:
#         with col1:
#             video_id = youtube_link.split("=")[1]
#             st.video(f"https://www.youtube.com/watch?v={video_id}", format="video/mp4")

#         # Prompt for extracting keywords
#         prompt = """From the transcript of the video, identify the 10 core topics/keyterms discussed and get them into a proper Python list []
#         separated by commas. Note that the transcript may contain grammatical/wording errors. Never get meaningless words."""

#         # Extract transcript using your actual function (replace with extract_transcript)
#         transcript = extract_transcript_details(youtube_link)

#         # Merge transcripts into a single transcript using merge_transcript
#         merged_transcript = merge_transcript(transcript, interval_minutes=5)

#         # Create a text transcript
#         merged_text = ""
#         for timestamp, text in merged_transcript.items():
#             merged_text += f"{timestamp}: {text}\n"

#         # Generate content based on the transcript (replace with your function)
#         if not st.session_state.gemini_response:
#             st.session_state.gemini_response = generate_gemini_content(merged_text, prompt)

#         with st.expander('Show Transcript'):
#             st.markdown(merged_text)

#         # Convert the result to a Python list
#         my_list = ast.literal_eval(st.session_state.gemini_response)

#         # Display multiselect box
#         options = st.multiselect("Select keywords", my_list, default=st.session_state.selected_keywords)
#         st.write("You selected:", options)

#         # Update session state with selected keywords
#         st.session_state.selected_keywords = options

#         if options:
#             # Prompt for concise explanation
#             prompt_explanation = f"""You are an assistant who can analyze the following YouTube video transcript: {merged_text}
#             and provide a summary of what the transcript says about the following keywords: {options}. Note that you should provide the
#             answers based on the transcript only."""
#             # Generate content for the explanation (replace with your function)
#             concise_explanation = generate_gemini_content(merged_text, prompt_explanation)
#             st.subheader("Concise Explanation:")
#             st.markdown(concise_explanation)
          
# #quest answering
# def main3():
#     st.title("YouTube Video Question Answering")

#     # Initialize session state variables
#     if 'selected_keywords' not in st.session_state:
#         st.session_state.selected_keywords = []

#     # Input for YouTube video link
#     youtube_link = st.text_input("Enter YouTube Video Link:")
#     col1, col2 = st.columns([5, 3])

#     if youtube_link:
#         with col1:
#             video_id = youtube_link.split("=")[1]
#             st.video(f"https://www.youtube.com/watch?v={video_id}", format="video/mp4")

#         # User query input
#         user_query = st.text_area("Enter your query:")

#         # Prompt for extracting keywords
#         prompt = f"""You are an assistant who can analyze video transcript and answer user query: '{user_query}' in 50 words,
#         based on the transcript of the video, you will never generate random/creative/not genuine responses on your own,
#         never talk out of the context."""

#         # Extract transcript using your actual function (replace with extract_transcript)
#         transcript = extract_transcript_details(youtube_link)

#         # Merge transcripts into a single transcript using merge_transcript
#         merged_transcript = merge_transcript(transcript, interval_minutes=5)

#         # Create a text transcript
#         merged_text = ""
#         for timestamp, text in merged_transcript.items():
#             merged_text += f"{timestamp}: {text}\n"

#         # Generate content based on the transcript and user query (replace with your function)
#         if user_query:
#             res = generate_gemini_content(merged_text, prompt, max_output_tokens=100)
#             st.subheader("Generated Response:")
#             st.markdown(f"**{res}**")

# #mp4 
# def main4():
#     # Streamlit UI
#     st.title("YouTube/MP4 Audio Transcriber and Summarizer")
#     st.markdown("""<style>.stButton > button {display: block;margin: 0 auto;}</style>""", unsafe_allow_html=True)

#     # Define output paths
#     output_path = '/tmp/amios'
    
#     # Create folder if it doesn't exist
#     if not os.path.exists(output_path):
#         os.makedirs(output_path)

#     # Initialize session state variables
#     if 'video_selected' not in st.session_state:
#         st.session_state['video_selected'] = False
#     if 'file_uploaded' not in st.session_state:
#         st.session_state['file_uploaded'] = False
#     if 'existing_video_selected' not in st.session_state:
#         st.session_state['existing_video_selected'] = False

#     # User inputs
#     video_url = st.text_input("Enter YouTube video URL:", disabled=st.session_state['video_selected'] or st.session_state['file_uploaded'] or st.session_state['existing_video_selected'])
#     uploaded_file = st.file_uploader("Or upload an MP4 file", type=["mp4"], disabled=st.session_state['video_selected'] or bool(video_url) or st.session_state['existing_video_selected'])
    
#     # List existing videos
#     existing_videos = [f for f in os.listdir(output_path) if f.endswith('.mp4')]
#     selected_video = st.selectbox("Or select an existing video", [""] + existing_videos, index=0, disabled=bool(video_url) or uploaded_file is not None)

#     # Update session state based on user input
#     if selected_video and selected_video != "":
#         st.session_state['existing_video_selected'] = True
#         st.session_state['video_selected'] = True
#         st.session_state['file_uploaded'] = False
#         video_url = None
#         uploaded_file = None
#     elif uploaded_file:
#         st.session_state['file_uploaded'] = True
#         st.session_state['video_selected'] = False
#         st.session_state['existing_video_selected'] = False
#         video_url = None
#     elif video_url:
#         st.session_state['file_uploaded'] = False
#         st.session_state['video_selected'] = False
#         st.session_state['existing_video_selected'] = False

#     # Display selected video
#     if selected_video and selected_video != "":
#         st.video(os.path.join(output_path, selected_video))
#     elif uploaded_file:
#         st.video(uploaded_file)
#     elif video_url:
#         video_id = video_url.split("v=")[-1] if "v=" in video_url else video_url.split("/")[-1].split("?")[0]
#         st.video(f"https://www.youtube.com/embed/{video_id}")

#     # Process video
#     if st.button("Transcribe"):
#         with st.spinner('Processing...'):
#             try:
#                 if video_url:
#                     downloaded_file, video_title = download_audio(video_url, output_path)
#                 elif uploaded_file:
#                     downloaded_file = os.path.join(output_path, uploaded_file.name)
#                     with open(downloaded_file, "wb") as f:
#                         f.write(uploaded_file.getbuffer())
#                     video_title = os.path.splitext(uploaded_file.name)[0]
#                 elif selected_video:
#                     downloaded_file = os.path.join(output_path, selected_video)
#                     video_title = os.path.splitext(selected_video)[0]
#                 else:
#                     st.error("Please enter a YouTube URL, upload an MP4 file, or select an existing video.")
#                     st.stop()

#                 # Check if the downloaded file exists
#                 if not os.path.exists(downloaded_file):
#                     st.error("Downloaded file does not exist.")
#                     st.stop()
#                 else:
#                     st.write(f"Downloaded file path: {downloaded_file}")

#                 wav_output_file = os.path.splitext(downloaded_file)[0] + '.wav'
#                 convert_to_wav(downloaded_file, wav_output_file)

#                 # Debug: Check if the WAV file was created successfully
#                 if os.path.exists(wav_output_file):
#                     st.write(f"WAV file path: {wav_output_file}")
#                 else:
#                     st.error(f"Converted WAV file does not exist at {wav_output_file}")
#                     st.stop()

#                 text = transcribe_audio(wav_output_file)
#                 if text:
#                     transcription_filename = f"{video_title}.txt".replace(" ", "_").replace("/", "_")
#                     file_path = os.path.join(output_path, transcription_filename)
#                     with open(file_path, "w") as file:
#                         file.write(text)
#                     st.success("Transcription completed successfully")
#                     st.write(f"Transcription file path: {file_path}")
#                     with st.expander("Transcript"):
#                         st.markdown(f"#### {video_title}\n\n{text}")

#                     # Generate summary using Gemini model
#                     prompt = """You are a YouTube transcript summarizer. You will take youtube video transcript and provide detailed
#                     summary about major points discussed within 250 words. Never mention the name of the person"""
#                     try:
#                         summary = generate_gemini_content(text, prompt)
#                         st.markdown('## **Summary**')
#                         st.markdown(summary)
#                     except Exception as e:
#                         st.error("Can't generate summary")
#             except PermissionError:
#                 st.error("Permission error: Unable to create or write to the specified directory.")
#             except FileNotFoundError as e:
#                 st.error(f"File not found error: {e}")
#             except Exception as e:
#                 st.error(f"An error occurred: {e}")
              
# selected_option = st.radio("Select an option:",
#       ["YouTube Video Transcript Summarizer", "YouTube Video Keyword Content Analyzer","YouTube Video Question Answering",
#        "YouTube/MP4 Audio Transcriber and Summarizer"])

# if selected_option == "YouTube Video Transcript Summarizer":
#     main1()
# elif selected_option == "YouTube Video Keyword Content Analyzer":
#     main2()
# elif selected_option == "YouTube Video Question Answering":
#     main3()
# elif selected_option == "YouTube/MP4 Audio Transcriber and Summarizer":
#     main4()











#Current
import streamlit as st
import re, os, ast, speech_recognition as sr
import google.generativeai as genai, google.generativeai as gem
from datetime import datetime, timedelta
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube
import time
from moviepy.editor import AudioFileClip
gem.configure(api_key='AIzaSyBbepUh8x3CqpkxNFnJ1IX0dFc0UNTwwbU')

# Custom CSS to display radio options
st.markdown("""
    <style>
    div[role="radiogroup"] > label > div {
        display: flex;
        flex-direction: row;
    }
    div[role="radiogroup"] > label > div > div {
        margin-right: 10px;
    }
    </style>
    <style>
    div[role="radiogroup"] {
        display: flex;
        flex-direction: row;
    }
    div[role="radiogroup"] > label {
        margin-right: 20px;
    }
    input[type="radio"]:div {
        background-color: white;
        border-color: lightblue;
    }
    </style>
""", unsafe_allow_html=True)

prompt="""You are Youtube video summarizer. You will be taking the transcript text and summarizing the entire video and providing
                    the overall summary get the major points discussed,never mention the name odf the person
                    within 300 words in following format. Please provide the summary of the text given here:"""

def get_video_duration(youtube_video_url):
    try:
        # Fetch video details using pytube
        yt = YouTube(youtube_video_url)
        duration = yt.length  # Duration in seconds

        # Convert duration to minutes and seconds
        minutes, seconds = divmod(duration, 60)
        duration_formatted = f"{minutes} minutes and {seconds} seconds"

        return duration_formatted

    except Exception as e:
        raise e

# Function to extract transcript details
def extract_transcript_details(youtube_video_url):
    try:
        # Extract video ID from the URL
        if "youtu.be" in youtube_video_url:
            video_id = youtube_video_url.split("/")[-1]
        else:
            video_id = youtube_video_url.split("=")[1]

        # Fetch the video transcript (you can specify the language)
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=("en",))  # English transcript

        # Initialize a dictionary to store timestamps and transcriptions
        transcript_dict = {}

        # Extract text content and clean data
        for t in transcript:
            cleaned_text = re.sub(r"[^a-zA-Z0-9-ışğöüçiIŞĞÖÜÇİ ]", "", t["text"])
            if cleaned_text:
                timestamp = t["start"]
                formatted_time = f"{int(timestamp // 3600):02d}:{int((timestamp // 60) % 60):02d}:{int(timestamp % 60):02d}"
                transcript_dict[formatted_time] = cleaned_text

        return transcript_dict

    except Exception as e:
        st.error("No Transcript found", icon = '❕')
        return None
      
# Function to merge transcript
def merge_transcript(transcript, interval_minutes=.5):
    try:
        # Initialize variables
        interval = timedelta(minutes=interval_minutes)
        merged_transcript = {}
        current_interval_start = datetime.strptime('00:00:00', '%H:%M:%S')
        current_interval_end = current_interval_start + interval
        current_text = ""

        for timestamp, text in transcript.items():
            time = datetime.strptime(timestamp, '%H:%M:%S')

            # If the current timestamp is within the current interval, append the text
            if time < current_interval_end:
                current_text += text + " "
            else:
                # Save the current interval text to the merged_transcript
                merged_transcript[current_interval_end.strftime('%H:%M:%S')] = current_text.strip()
                # Move to the next interval
                while time >= current_interval_end:
                    current_interval_start = current_interval_end
                    current_interval_end = current_interval_start + interval
                # Start the new interval with the current text
                current_text = text + " "

        # Add the last interval if there's remaining text
        if current_text:
            merged_transcript[current_interval_end.strftime('%H:%M:%S')] = current_text.strip()

        # Ensure the last key matches the last key from the input transcript
        last_key_input = list(transcript.keys())[-1]
        last_key_output = list(merged_transcript.keys())[-1]
        if last_key_input != last_key_output:
            merged_transcript[last_key_input] = merged_transcript.pop(last_key_output)

        return merged_transcript

    except Exception as e:
        st.error(f"Unable to continue transcript process")
        return None

def generate_gemini_content(transcript_text, prompt, max_output_tokens=None):
  with st.spinner('Summarizing...'):
    try:
        os.environ['GOOGLE_API_KEY'] = 'AIzaSyBbepUh8x3CqpkxNFnJ1IX0dFc0UNTwwbU'  # Replace with your API key
        genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

        models = [m for m in genai.list_models() if "text-bison" in m.name]
        model = models[0].name

        if max_output_tokens is None:
            max_output_tokens = 300

        # Generate text using the specified model
        completion = genai.generate_text(
            model=model,
            prompt=transcript_text + prompt,
            temperature=0.1,
            max_output_tokens=max_output_tokens
        )

        output = completion.result.split('\n')
        response = "\n".join(output)
        return response
    except Exception as e:
        st.exception("Can't generate summary")
        return ""
      
# def delete_file(file_path):
#     try:
#         os.remove(file_path)
#         print(f"Deleted file: {file_path}")
#     except Exception as e:
#         print(f"An error occurred while deleting the file: {e}")

output_path = '/tmp/amios'
if not os.path.exists(output_path):
    os.makedirs(output_path)
  
def convert_to_wav(input_file, output_file):
  with st.spinner('Converting audio to WAV...'):
    try:
        audio_clip = AudioFileClip(input_file)
        audio_clip.write_audiofile(output_file)
        # st.caption(f"Converted {input_file} to {output_file}")
    except Exception as e:
        st.exception(f"An error occurred while converting to WAV: {e}")

def download_audio(video_url, output_path):
  with st.spinner('Downloading audio...'):
    try:
        yt = YouTube(video_url)
        video_title = yt.title
        audio_stream = yt.streams.filter(only_audio=True).first()
        downloaded_file = audio_stream.download(output_path=output_path)
        # st.caption(f"Audio downloaded successfully to {downloaded_file}")
        return downloaded_file, video_title
    except Exception as e:
        st.exception(f"An error occurred: {e}")
        return None, None

def transcribe_audio(file_path, language='en-US', retries=3):
  with st.spinner('Transcribing audio...'):
    r = sr.Recognizer()
    for attempt in range(retries):
        try:
            with sr.AudioFile(file_path) as source:
                audio_text = r.record(source)
                text = r.recognize_google(audio_text, language=language)
                return text
        except sr.UnknownValueError:
            st.write("Could not understand audio")
            return ""
        except sr.RequestError as req:
            st.error(f"Error fetching results from Google Web Speech API: {req}")
            if attempt < retries - 1:
                time.sleep(2)  # Wait for 2 seconds before retrying
            else:
                st.error(f"Transcription failed after {retries} attempts.")
              
#summarizer
def main1():
    st.markdown(
            """
            <style>
            .stButton > button {
                display: block;
                margin: 0 auto;
            }
            </style>
            """, unsafe_allow_html=True)
    # Streamlit UI
    st.title("YouTube Video Transcript Summarizer")
    # Initialize session state variables
    if 'youtube_link' not in st.session_state:
        st.session_state.youtube_link = ""
    if 'interval_minutes' not in st.session_state:
        st.session_state.interval_minutes = 5
    if 'transcript_extracted' not in st.session_state:
        st.session_state.transcript_extracted = False
    if 'merged_text' not in st.session_state:
        st.session_state.merged_text = ""
    if 'show_transcript' not in st.session_state:
        st.session_state.show_transcript = False

    youtube_link = st.text_input("Enter YouTube Video Link:")

    if youtube_link:
        st.session_state.youtube_link = youtube_link

    col1, col2 = st.columns([5, 3])

    if st.session_state.youtube_link:
        with col1:
            # video_id = st.session_state.youtube_link.split("=")[1]
            # st.video(f"https://www.youtube.com/watch?v={video_id}", format="video/mp4")
            video_id = youtube_link.split("v=")[-1] if "v=" in youtube_link else youtube_link.split("/")[-1].split("?")[0]
            st.video(f"https://www.youtube.com/embed/{video_id}")
          
        dur = get_video_duration(st.session_state.youtube_link)
        st.info(dur)
        # interval_minutes = st.slider("Select interval in minutes", min_value=1, max_value=10, value=5, step=1)
        # if interval_minutes:
        #     st.session_state.interval_minutes = interval_minutes

        if st.button('Get Summary'):
            with st.spinner("Extracting transcript..."):
                try:
                    transcript_text = extract_transcript_details(st.session_state.youtube_link)
                    if transcript_text:
                        st.success("Transcript extracted successfully!")
                        try:
                            # merged_transcript = merge_transcript(transcript_text, st.session_state.interval_minutes)
                            merged_transcript = merge_transcript(transcript_text, 4)
                            if merged_transcript:
                                merged_text = ""
                                for timestamp, text in merged_transcript.items():
                                    merged_text += f"{timestamp}: {text}\n"
                                st.session_state.merged_text = merged_text
                                st.session_state.transcript_extracted = True
                                if st.session_state.transcript_extracted:
                                    with st.expander('Show Transcript'):
                                        st.session_state.show_transcript = True
                                        if st.session_state.show_transcript:
                                            st.write(st.session_state.merged_text)
                                try:
                                    # res = generate_gemini_content(str(merged_text),prompt)
                                    # st.write(res)
                                    o = gem.GenerativeModel('gemini-1.5-pro-latest')
                                    explanation = o.generate_content(f"""You are Youtube video summarizer. You will be taking the transcript text and summarizing
                                    the entire video and providing the overall summary get the major points discussed,never mention the name odf the person
                    within 300 words in following format. Please provide the summary of the text given here:{merged_text}""")
                                    st.markdown('## **Summary**')
                                    st.write(explanation.text)
                                except:
                                    st.exception('Failed to get summary')
                        except Exception as e:
                            st.exception(f"Failed to merge transcript: {e}")
                except Exception as e:
                    st.exception(f"Failed to extract transcript: {e}")

# keyword extractor -palm
# def main2():
#     st.title("YouTube Video Keyword Content Analyzer")

#     # Initialize session state variables
#     if 'selected_keywords' not in st.session_state:
#         st.session_state.selected_keywords = []
#     if 'gemini_response' not in st.session_state:
#         st.session_state.gemini_response = ""
#     if 'previous_youtube_link' not in st.session_state:
#         st.session_state.previous_youtube_link = ""

#     # Input for YouTube video link
#     youtube_link = st.text_input("Enter YouTube Video Link:")

#     # Clear selected keywords if a new YouTube link is entered
#     if youtube_link != st.session_state.previous_youtube_link:
#         st.session_state.selected_keywords = []
#         st.session_state.gemini_response = ""
#         st.session_state.previous_youtube_link = youtube_link

#     col1, col2 = st.columns([5, 3])

#     if youtube_link:
#         with col1:
#             video_id = youtube_link.split("v=")[-1] if "v=" in youtube_link else youtube_link.split("/")[-1].split("?")[0]
#             st.video(f"https://www.youtube.com/embed/{video_id}")

#         # Prompt for extracting keywords
#         prompt = """From the transcript of the video, identify the 10 core topics/keyterms discussed and get them into a proper Python list []
#         separated by commas. Note that the transcript may contain grammatical/wording errors. Never get meaningless words."""

#         # Extract transcript using your actual function (replace with extract_transcript)
#         transcript = extract_transcript_details(youtube_link)

#         # Merge transcripts into a single transcript using merge_transcript
#         merged_transcript = merge_transcript(transcript, interval_minutes=5)

#         # Create a text transcript
#         merged_text = ""
#         for timestamp, text in merged_transcript.items():
#             merged_text += f"{timestamp}: {text}\n"

#         # Generate content based on the transcript (replace with your function)
#         if not st.session_state.gemini_response:
#             st.session_state.gemini_response = generate_gemini_content(merged_text, prompt)

#         with st.expander('Show Transcript'):
#             st.markdown(merged_text)

#         # Convert the result to a Python list
#         my_list = ast.literal_eval(st.session_state.gemini_response)

#         # Display multiselect box
#         options = st.multiselect("Select keywords", my_list, default=st.session_state.selected_keywords)
#         st.write("You selected:", options)

#         # Update session state with selected keywords
#         st.session_state.selected_keywords = options

#         if options:
#             # Prompt for concise explanation
#             prompt_explanation = f"""You are an assistant who can analyze the following YouTube video transcript: {merged_text}
#             and provide a summary of what the transcript says about the following keywords: {options} in 250 words. Note that you should provide the
#             answers based on the transcript only."""
#             # Generate content for the explanation (replace with your function)
#             concise_explanation = generate_gemini_content(merged_text, prompt_explanation)
#             st.subheader("Concise Explanation:")
#             st.markdown(concise_explanation)

#gemini
def main2():
    st.title("YouTube Video Keyword Content Analyzer")

    # Initialize session state variables
    if 'selected_keywords' not in st.session_state:
        st.session_state.selected_keywords = []
    if 'gemini_response' not in st.session_state:
        st.session_state.gemini_response = ""
    if 'previous_youtube_link' not in st.session_state:
        st.session_state.previous_youtube_link = ""

    # Input for YouTube video link
    youtube_link = st.text_input("Enter YouTube Video Link:")

    # Clear selected keywords if a new YouTube link is entered
    if youtube_link != st.session_state.previous_youtube_link:
        st.session_state.selected_keywords = []
        st.session_state.gemini_response = ""
        st.session_state.previous_youtube_link = youtube_link

    col1, col2 = st.columns([5, 3])

    if youtube_link:
        with col1:
            video_id = youtube_link.split("v=")[-1] if "v=" in youtube_link else youtube_link.split("/")[-1].split("?")[0]
            st.video(f"https://www.youtube.com/embed/{video_id}")

        # Prompt for extracting keywords
        # prompt = """From the transcript of the video, identify the 10 core topics/keyterms discussed and get them into a proper Python list []
        # separated by commas. Note that the transcript may contain grammatical/wording errors. Never get meaningless words."""

        # Extract transcript using your actual function (replace with extract_transcript)
        transcript = extract_transcript_details(youtube_link)

        # Merge transcripts into a single transcript using merge_transcript
        merged_transcript = merge_transcript(transcript, interval_minutes=5)

        # Create a text transcript
        merged_text = ""
        for timestamp, text in merged_transcript.items():
            merged_text += f"{timestamp}: {text}\n"

        # Generate content based on the transcript (replace with your function)
        if not st.session_state.gemini_response:
            # st.session_state.gemini_response = generate_gemini_content(merged_text, prompt)
            o = gem.GenerativeModel('gemini-1.5-pro-latest')
            st.session_state.gemini_response = o.generate_content(f"""From the transcript of the video :{merged_text}, identify the 10 core topics/keyterms 
            discussed and get them into a proper clean pure Python list separated by commas,your response shall not contain
            things like ```python and ``` but shall have [].""").text
              
        with st.expander('Show Transcript'):
            st.markdown(merged_text)

        # Convert the result to a Python list
        my_list = ast.literal_eval(st.session_state.gemini_response)

        # Display multiselect box
        options = st.multiselect("Select keywords", my_list, default=st.session_state.selected_keywords)
        st.write("You selected:", options)

        # Update session state with selected keywords
        st.session_state.selected_keywords = options

        # gemini 1.5 
        if options:
            # Prompt for concise explanation
            prompt_explanation = f"""You are an assistant who can analyze the following YouTube video transcript: {merged_text}
            and provide a summary of what the transcript says about the following keywords: {options}. Note that you should provide the
            answers based on the transcript only."""
            # Generate content for the explanation using the provided gem.GenerativeModel implementation
            o = gem.GenerativeModel('gemini-1.5-pro-latest')
            concise_explanation = o.generate_content(prompt_explanation)
            st.subheader("Concise Explanation:")
            st.markdown(concise_explanation.text)

        # txt bison  
        # if options:
        #     # Prompt for concise explanation
        #     prompt_explanation = f"""You are an assistant who can analyze the following YouTube video transcript: {merged_text}
        #     and provide a summary of what the transcript says about the following keywords: {options} in 250 words. Note that you should provide the
        #     answers based on the transcript only."""
        #     # Generate content for the explanation (replace with your function)
        #     concise_explanation = generate_gemini_content(merged_text, prompt_explanation)
        #     st.subheader("Concise Explanation:")
        #     st.markdown(concise_explanation)
      
                      
#quest answering
def main3():
    st.title("YouTube Video Question Answering")

    # Initialize session state variables
    if 'selected_keywords' not in st.session_state:
        st.session_state.selected_keywords = []

    # Input for YouTube video link
    youtube_link = st.text_input("Enter YouTube Video Link:")
    col1, col2 = st.columns([5, 3])

    if youtube_link:
        with col1:
            # video_id = youtube_link.split("=")[1]
            # st.video(f"https://www.youtube.com/watch?v={video_id}", format="video/mp4")
            video_id = youtube_link.split("v=")[-1] if "v=" in youtube_link else youtube_link.split("/")[-1].split("?")[0]
            st.video(f"https://www.youtube.com/embed/{video_id}")

        # User query input
        user_query = st.text_area("Enter your query:")

        # Prompt for extracting keywords
        # prompt = f"""You are an assistant who can analyze video transcript and answer user query: '{user_query}' in 50 words,
        # based on the transcript of the video, you will never generate random/creative/not genuine responses on your own,
        # never talk out of the context."""

        # Extract transcript using your actual function (replace with extract_transcript)
        transcript = extract_transcript_details(youtube_link)

        # Merge transcripts into a single transcript using merge_transcript
        merged_transcript = merge_transcript(transcript, interval_minutes=5)

        # Create a text transcript
        merged_text = ""
        for timestamp, text in merged_transcript.items():
            merged_text += f"{timestamp}: {text}\n"

        # Generate content based on the transcript and user query (replace with your function)
        if user_query:
            # res = generate_gemini_content(merged_text, prompt, max_output_tokens=240)
            
            # st.markdown(f"**{res}**")
            o = gem.GenerativeModel('gemini-1.5-pro-latest')
            res = o.generate_content(f"""You are an assistant who can analyze video transcript and answer user query: '{user_query}' in 100 words,
        based on the transcript of the video :{merged_text}, you will never generate random/creative/not genuine responses on your own,
        never talk out of the context.""").text
            st.subheader("Generated Response:")
            st.markdown(res)

#mp4 - palm
# def main4():
#     # Streamlit UI
#     st.title("YouTube/MP4 Audio Transcriber and Summarizer")
#     st.markdown("""<style>.stButton > button {display: block;margin: 0 auto;}</style>""", unsafe_allow_html=True)

#     # Define output paths
#     output_path = '/tmp/amios'
    
#     # Create folder if it doesn't exist
#     if not os.path.exists(output_path):
#         os.makedirs(output_path)

#     # Initialize session state variables
#     if 'video_selected' not in st.session_state:
#         st.session_state['video_selected'] = False
#     if 'file_uploaded' not in st.session_state:
#         st.session_state['file_uploaded'] = False
#     if 'existing_video_selected' not in st.session_state:
#         st.session_state['existing_video_selected'] = False

#     # User inputs
#     video_url = st.text_input("Enter YouTube video URL:", disabled=st.session_state['video_selected'] or st.session_state['file_uploaded'] or st.session_state['existing_video_selected'])
#     uploaded_file = st.file_uploader("Or upload an MP4 file", type=["mp4"], disabled=st.session_state['video_selected'] or bool(video_url) or st.session_state['existing_video_selected'])
    
#     # List existing videos
#     existing_videos = [f for f in os.listdir(output_path) if f.endswith('.mp4')]
#     selected_video = st.selectbox("Or select an existing video", [""] + existing_videos, index=0, disabled=bool(video_url) or uploaded_file is not None)

#     # Update session state based on user input
#     if selected_video and selected_video != "":
#         st.session_state['existing_video_selected'] = True
#         st.session_state['video_selected'] = True
#         st.session_state['file_uploaded'] = False
#         video_url = None
#         uploaded_file = None
#     elif uploaded_file:
#         st.session_state['file_uploaded'] = True
#         st.session_state['video_selected'] = False
#         st.session_state['existing_video_selected'] = False
#         video_url = None
#     elif video_url:
#         st.session_state['file_uploaded'] = False
#         st.session_state['video_selected'] = False
#         st.session_state['existing_video_selected'] = False

#     # Display selected video
#     if selected_video and selected_video != "":
#         st.video(os.path.join(output_path, selected_video))
#     elif uploaded_file:
#         st.video(uploaded_file)
#     elif video_url:
#         video_id = video_url.split("v=")[-1] if "v=" in video_url else video_url.split("/")[-1].split("?")[0]
#         st.video(f"https://www.youtube.com/embed/{video_id}")

#     # Process video
#     if st.button("Transcribe"):
#         with st.spinner('Processing...'):
#             try:
#                 if video_url:
#                     downloaded_file, video_title = download_audio(video_url, output_path)
#                 elif uploaded_file:
#                     downloaded_file = os.path.join(output_path, uploaded_file.name)
#                     with open(downloaded_file, "wb") as f:
#                         f.write(uploaded_file.getbuffer())
#                     video_title = os.path.splitext(uploaded_file.name)[0]
#                 elif selected_video:
#                     downloaded_file = os.path.join(output_path, selected_video)
#                     video_title = os.path.splitext(selected_video)[0]
#                 else:
#                     st.error("Please enter a YouTube URL, upload an MP4 file, or select an existing video.")
#                     st.stop()

#                 # Check if the downloaded file exists
#                 if not os.path.exists(downloaded_file):
#                     st.error("Downloaded file does not exist.")
#                     st.stop()
#                 # else:
#                 #     st.caption(f"Downloaded file path: {downloaded_file}")

#                 wav_output_file = os.path.splitext(downloaded_file)[0] + '.wav'
#                 convert_to_wav(downloaded_file, wav_output_file)

#                 # Debug: Check if the WAV file was created successfully
#                 # if os.path.exists(wav_output_file):
#                     # st.caption(f"WAV file path: {wav_output_file}")
#                 # else:
#                     # st.error(f"Converted WAV file does not exist at {wav_output_file}")
#                     # st.stop()

#                 text = transcribe_audio(wav_output_file)
#                 if text:
#                     transcription_filename = f"{video_title}.txt".replace(" ", "_").replace("/", "_")
#                     file_path = os.path.join(output_path, transcription_filename)
#                     with open(file_path, "w") as file:
#                         file.write(text)
#                     # st.success("Transcription completed successfully")
#                     # st.caption(f"Transcription file path: {file_path}")
#                     with st.expander("Transcript"):
#                         st.markdown(f"#### {video_title}\n\n{text}")

#                     # Generate summary using Gemini model
#                     prompt = """You are a YouTube transcript summarizer. You will take youtube video transcript and provide 
#                     summary about major points discussed within 250 words. Never mention the name of the person"""
#                     try:
#                         summary = generate_gemini_content(text, prompt)
#                         st.markdown('## **Summary**')
#                         st.markdown(summary)
#                     except Exception as e:
#                         st.exception("Can't generate summary")
#             except PermissionError:
#                 st.error("Permission error: Unable to create or write to the specified directory.")
#             except FileNotFoundError as e:
#                 st.error(f"File not found error: {e}")
#             except Exception as e:
#                 st.exception(f"An error occurred: {e}")


# #mp4 -gemini
def main4():
    # Streamlit UI
    st.title("YouTube/MP4 Audio Transcriber and Summarizer")
    st.markdown("""<style>.stButton > button {display: block;margin: 0 auto;}</style>""", unsafe_allow_html=True)

    # Define output paths
    output_path = '/tmp/amios'

    # Create folder if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Initialize session state variables
    if 'video_selected' not in st.session_state:
        st.session_state['video_selected'] = False
    if 'file_uploaded' not in st.session_state:
        st.session_state['file_uploaded'] = False
    if 'existing_video_selected' not in st.session_state:
        st.session_state['existing_video_selected'] = False

    # User inputs
    video_url = st.text_input("Enter YouTube video URL:", disabled=st.session_state['video_selected'] or st.session_state['file_uploaded'] or st.session_state['existing_video_selected'])
    uploaded_file = st.file_uploader("Or upload an MP4 file", type=["mp4"], disabled=st.session_state['video_selected'] or bool(video_url) or st.session_state['existing_video_selected'])

    # List existing videos
    existing_videos = [f for f in os.listdir(output_path) if f.endswith('.mp4')]
    selected_video = st.selectbox("Or select an existing video", [""] + existing_videos, index=0, disabled=bool(video_url) or uploaded_file is not None)

    # Update session state based on user input
    if selected_video and selected_video != "":
        st.session_state['existing_video_selected'] = True
        st.session_state['video_selected'] = True
        st.session_state['file_uploaded'] = False
        video_url = None
        uploaded_file = None
    elif uploaded_file:
        st.session_state['file_uploaded'] = True
        st.session_state['video_selected'] = False
        st.session_state['existing_video_selected'] = False
        video_url = None
    elif video_url:
        st.session_state['file_uploaded'] = False
        st.session_state['video_selected'] = False
        st.session_state['existing_video_selected'] = False

    # Display selected video
    if selected_video and selected_video != "":
        st.video(os.path.join(output_path, selected_video))
    elif uploaded_file:
        st.video(uploaded_file)
    elif video_url:
        video_id = video_url.split("v=")[-1] if "v=" in video_url else video_url.split("/")[-1].split("?")[0]
        st.video(f"https://www.youtube.com/embed/{video_id}")

    # Process video
    if st.button("Transcribe"):
        with st.spinner('Processing...'):
            try:
                if video_url:
                    downloaded_file, video_title = download_audio(video_url, output_path)
                elif uploaded_file:
                    downloaded_file = os.path.join(output_path, uploaded_file.name)
                    with open(downloaded_file, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    video_title = os.path.splitext(uploaded_file.name)[0]
                elif selected_video:
                    downloaded_file = os.path.join(output_path, selected_video)
                    video_title = os.path.splitext(selected_video)[0]
                else:
                    st.error("Please enter a YouTube URL, upload an MP4 file, or select an existing video.")
                    st.stop()

                # Check if the downloaded file exists
                if not os.path.exists(downloaded_file):
                    st.error("Downloaded file does not exist.")
                    st.stop()
                # else:
                    # st.caption(f"Downloaded file path: {downloaded_file}")

                wav_output_file = os.path.splitext(downloaded_file)[0] + '.wav'
                convert_to_wav(downloaded_file, wav_output_file)

                # Debug: Check if the WAV file was created successfully
                # if os.path.exists(wav_output_file):
                    # st.caption(f"WAV file path: {wav_output_file}")
                # else:
                    # st.error(f"Converted WAV file does not exist at {wav_output_file}")
                    # st.stop()

                text = transcribe_audio(wav_output_file)
                if text:
                    transcription_filename = f"{video_title}.txt".replace(" ", "_").replace("/", "_")
                    file_path = os.path.join(output_path, transcription_filename)
                    with open(file_path, "w") as file:
                        file.write(text)
                    # st.success("Transcription completed successfully")
                    # st.caption(f"Transcription file path: {file_path}")
                    with st.expander("Transcript"):
                        st.markdown(f"#### {video_title}\n\n{text}")

                    try:
                        o = gem.GenerativeModel('gemini-1.5-pro-latest')
                        summary = o.generate_content(f"""You are a YouTube transcript summarizer. You will take youtube video 
                        transcript and provide summary about major points discussed within 250 words. 
                        Never mention the name of the person, here is the transcript:{text}""")
                        st.markdown('## **Summary**')
                        st.markdown(summary.text)
                    except Exception as e:
                        st.exception("Can't generate summary")
            except PermissionError:
                st.error("Permission error: Unable to create or write to the specified directory.")
            except FileNotFoundError as e:
                st.error(f"File not found error: {e}")
            except Exception as e:
                st.exception(f"An error occurred: {e}")

selected_option = st.radio("Select an option:",
      ["YouTube Video Transcript Summarizer", "YouTube Video Keyword Content Analyzer","YouTube Video Question Answering",
       "YouTube/MP4 Audio Transcriber and Summarizer"])

if selected_option == "YouTube Video Transcript Summarizer":
    main1()
elif selected_option == "YouTube Video Keyword Content Analyzer":
    main2()
elif selected_option == "YouTube Video Question Answering":
    main3()
elif selected_option == "YouTube/MP4 Audio Transcriber and Summarizer":
    main4()
