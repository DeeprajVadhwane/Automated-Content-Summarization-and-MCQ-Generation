import streamlit as st
import validators
import os
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from pytube.exceptions import PytubeError

# Set up the Streamlit app
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')

# Get the Groq API Key and URL (YouTube or website) to be summarized
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

generic_url = st.text_input("Enter URL (YouTube or Website)", label_visibility="collapsed")

# Language selection for transcripts
language_options = {
    "en": "English",
    "hi": "Hindi",
    # Add more languages as needed
}

selected_language = st.selectbox("Select Transcript Language", options=list(language_options.keys()), format_func=lambda x: language_options[x])

# Check if the API key is provided
if not groq_api_key.strip():
    st.error("Please enter your Groq API Key.")
else:
    # Optionally set as an environment variable (or pass it directly)
    os.environ["GROQ_API_KEY"] = groq_api_key

    # Initialize the model
    llm = ChatGroq(model="Gemma-7b-It", groq_api_key=groq_api_key)

    prompt_template = """
    Provide a summary of the following content in 300 words:
    Content:{text}
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

    if st.button("Summarize the Content from YT or Website"):
        # Validate all the inputs
        if not generic_url.strip():
            st.error("Please provide a valid URL.")
        elif not validators.url(generic_url):
            st.error("Please enter a valid URL. It can be a YouTube video URL or a website URL.")
        else:
            try:
                with st.spinner("Loading content..."):
                    # Loading the website or YouTube video data
                    if "youtube.com" in generic_url:
                        loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True, languages=[selected_language])
                    else:
                        loader = UnstructuredURLLoader(
                            urls=[generic_url],
                            ssl_verify=False,
                            headers={"User-Agent": "Mozilla/5.0"}
                        )
                    
                    docs = loader.load()

                    # Chain for Summarization
                    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                    output_summary = chain.run(docs)

                    st.success(output_summary)

            except PytubeError as e:
                st.error(f"An error occurred with Pytube: {str(e)}. Please check the video URL or try again later.")
            except Exception as e:
                if "No transcripts were found" in str(e):
                    st.warning(f"No transcripts found in {selected_language}. Available languages: Hindi (auto-generated).")
                else:
                    st.error(f"An error occurred: {str(e)}")
