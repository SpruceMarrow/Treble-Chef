Project Description:

Treble Chef is an AI-based creative tool designed to generate original melodies and matching lyrics from scratch. Combining natural language processing (NLP) with deep learning techniques in music generation, the system serves as an intelligent co-writer for musicians, composers, and content creators. LyricMuse doesn’t require any musical input—just a theme, mood, or genre—and it outputs a fully-formed song idea with both melody and lyrical content.

✨ Key Features:

Melody Generation: Uses a neural network trained on a large dataset of music (e.g., MIDI files) to generate coherent, genre-specific melodies in various time signatures, keys, and styles.

Lyric Composition: Implements transformer-based language models (like GPT) fine-tuned on lyrical datasets to create original lyrics that align with the user’s chosen theme, tone, and structure (e.g., verse, chorus, bridge).

Melody-Lyric Alignment: Ensures rhythmic and emotional synchronization between the generated melody and lyrics using attention mechanisms and alignment scoring.

User Input Options:

Theme or topic (e.g., love, heartbreak, empowerment)

Audio Rendering: Converts melodies into playable audio (e.g., using MIDI synthesizers or neural audio synthesis) so users can hear a sample output.

🧠 Tech Stack:

Melody Generation: LSTM/Transformer models trained on MIDI sequences (e.g., MusicVAE, MuseNet, or custom architectures)

Lyrics Generation: GPT-style transformer model fine-tuned on song lyrics corpus

Audio Synthesis: FluidSynth, Magenta’s Performance RNN, or Jukebox (for higher fidelity outputs)

Frontend (Optional): Web interface built with React or Python GUI for non-technical users

🎯 Use Cases:

Songwriters seeking inspiration

Indie musicians prototyping ideas

Game developers or filmmakers needing quick, original music

Educational tools for music and songwriting

🚀 Potential Add-ons:

Style transfer (e.g., rewrite lyrics in the style of a specific artist)

Vocal melody generation for vocalists

Multi-language lyric generation

Chord progression and harmony layering
___________________________________________

Python Libraries to be installed:
streamlit, muspy, numpy, torch, midi2audio, pretty_midi, soundfile, google-genai

Replace the 'YOUR_API_KEY' variable in the TC.py file with your Gemini API Key


To run the app, run the following command in your CMD (In your working directory):
**python -m streamlit run TC.py**

