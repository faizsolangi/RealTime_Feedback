import streamlit as st
from langchain.llms import OpenAI
from crewai import Agent, Task, Crew
import assemblyai as aai
import requests
import os
import time
import threading
import pandas as pd

# Set up Streamlit page
st.title("Real-Time HubSpot Call Analysis Dashboard")
st.header("Instant Feedback for Sales Representatives")
st.markdown("Process live HubSpot call data with 3-5 second feedback delay and log to CRM.")

# Initialize APIs
llm = OpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
HUBSPOT_API_KEY = os.getenv("HUBSPOT_API_KEY")
CONTACT_ID = os.getenv("HUBSPOT_CONTACT_ID", "CONTACT_ID")  # Replace with Faiz Ahmad's contact ID

# Session state
if "transcript" not in st.session_state:
    st.session_state.transcript = ""
if "running" not in st.session_state:
    st.session_state.running = False
if "results" not in st.session_state:
    st.session_state.results = {"sentiment": {}, "key_phrases": [], "feedback": []}

# CrewAI agents
sentiment_analyst = Agent(
    role="Sentiment Analyst",
    goal="Analyze sentiment of call transcript in real-time",
    backstory="Expert in real-time NLP",
    verbose=True,
    llm=llm
)
feedback_generator = Agent(
    role="Feedback Generator",
    goal="Extract key phrases and provide instant feedback",
    backstory="Sales coach for real-time analysis",
    verbose=True,
    llm=llm
)

# Analyze transcript chunk
def analyze_chunk(chunk):
    try:
        sentiment_task = Task(
            description=f"Analyze sentiment: {chunk}",
            agent=sentiment_analyst,
            expected_output="Dictionary with positive, negative, neutral scores"
        )
        feedback_task = Task(
            description=f"Extract key phrases and provide feedback: {chunk}",
            agent=feedback_generator,
            expected_output="Dictionary with key_phrases (list) and feedback (list)"
        )
        crew = Crew(agents=[sentiment_analyst, feedback_generator], tasks=[sentiment_task, feedback_task], verbose=2)
        result = crew.kickoff()
        return eval(result.tasks_output[0].raw), eval(result.tasks_output[1].raw)
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        return {"positive": 0, "negative": 0, "neutral": 0}, {"key_phrases": [], "feedback": []}

# Log to HubSpot
def log_to_hubspot(transcript, results):
    try:
        response = requests.post(
            "https://api.hubapi.com/crm/v3/objects/engagements",
            headers={"Authorization": f"Bearer {HUBSPOT_API_KEY}"},
            json={
                "properties": {
                    "hs_call_notes": f"{transcript}\n\nAnalysis:\nSentiment: {results['sentiment']}\nKey Phrases: {results['key_phrases']}\nFeedback: {results['feedback']}",
                    "hs_call_to_number": "+1 (582) 203-8284",
                    "hs_call_disposition": "Cold Call",
                    "hs_timestamp": "2025-08-20T04:17:00Z"
                },
                "associations": [{"to": {"id": CONTACT_ID}, "types": [{"associationCategory": "HUBSPOT_DEFINED", "associationTypeId": 5}]}]
            }
        )
        return response.status_code == 201
    except Exception as e:
        st.error(f"HubSpot logging error: {str(e)}")
        return False

# Real-time transcription with AssemblyAI
def transcribe_real_time():
    transcriber = aai.RealtimeTranscriber(
        on_data=lambda data: (
            setattr(st.session_state, "transcript", st.session_state.transcript + data.text + " "),
            st.session_state.results.update(analyze_chunk(data.text)),
            st.rerun()
        ),
        on_error=lambda error: st.error(f"Transcription error: {error}")
    )
    transcriber.connect()
    # Simulate HubSpot call audio (replace with SDK stream)
    try:
        transcriber.stream_file("sample_call.wav")  # Replace with HubSpot SDK audio stream
    except FileNotFoundError:
        st.warning("Audio file not found. Simulating with sample transcript...")
        for chunk in [
            "Agent: Hello, this is a test discovery call from Test Corp. We provide customized marketing solutions to enhance your business growth. Can you share your current marketing challenges?",
            "Prospect: We’re struggling with lead generation and need better ROI on our campaigns.",
            "Agent: That’s a common challenge. Our solutions can optimize your campaigns using data-driven strategies. Can you describe your target audience?",
            "Prospect: Mostly small businesses. I’d like to know more about your pricing and implementation process.",
            "Agent: Let’s schedule a follow-up to discuss pricing and tailor a plan. Does next week work?",
            "Prospect: Yes, let’s do Tuesday."
        ]:
            if not st.session_state.running:
                break
            st.session_state.transcript += chunk + "\n"
            sentiment, feedback = analyze_chunk(chunk)
            st.session_state.results["sentiment"] = sentiment
            st.session_state.results["key_phrases"].extend(feedback.get("key_phrases", []))
            st.session_state.results["feedback"].extend(feedback.get("feedback", []))
            st.rerun()
            time.sleep(3)
    transcriber.close()
    st.session_state.running = False

# UI controls
col1, col2 = st.columns(2)
with col1:
    if st.button("Start Real-Time Transcription and Analysis"):
        if not st.session_state.running:
            st.session_state.running = True
            st.session_state.transcript = ""
            st.session_state.results = {"sentiment": {}, "key_phrases": [], "feedback": []}
            threading.Thread(target=transcribe_real_time, daemon=True).start()
with col2:
    if st.button("Stop Analysis"):
        st.session_state.running = False

if st.button("Log to HubSpot"):
    if log_to_hubspot(st.session_state.transcript, st.session_state.results):
        st.success("Logged to HubSpot!")
    else:
        st.error("Failed to log to HubSpot.")

# Display transcript
st.subheader("Live Transcript")
st.text_area("Transcript", st.session_state.transcript, height=200, disabled=True)

# Display results
st.subheader("Real-Time Analysis Results")
if st.session_state.results["sentiment"]:
    st.write("**Sentiment Analysis**")
    col1, col2, col3 = st.columns(3)
    col1.metric("Positive", f"{st.session_state.results['sentiment'].get('positive', 0):.2f}")
    col2.metric("Negative", f"{st.session_state.results['sentiment'].get('negative', 0):.2f}")
    col3.metric("Neutral", f"{st.session_state.results['sentiment'].get('neutral', 0):.2f}")
    # Sentiment chart
    st.write("**Sentiment Distribution**")
    sentiment_df = pd.DataFrame([st.session_state.results["sentiment"]], index=["Sentiment"])
    sentiment_df = sentiment_df.rename(columns={"positive": "Positive", "negative": "Negative", "neutral": "Neutral"})
    st.bar_chart(sentiment_df)

if st.session_state.results["key_phrases"]:
    st.write("**Key Phrases**")
    for phrase in st.session_state.results["key_phrases"]:
        st.write(f"- {phrase}")

if st.session_state.results["feedback"]:
    st.write("**Feedback for Improvement**")
    for item in st.session_state.results["feedback"]:
        st.write(f"- {item}")