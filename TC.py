import streamlit as st
import muspy, glob, json, os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
from midi2audio import FluidSynth
import pretty_midi
import soundfile as sf
from google import genai

# --- PAGE CONFIG ---
st.set_page_config(page_title="Treble Chef", page_icon="🎶", layout="centered")

# --- CSS Background ---
st.markdown("""
    <style>
    body {
      margin: 0;
      padding: 0;
      font-family: Arial, sans-serif;
      height: 100vh;
      background: linear-gradient(270deg, #ff6ec4, #7873f5, #42e695, #ff6ec4);
      background-size: 800% 800%;
      animation: gradientBG 12s ease infinite;
      color: white;
    }
    @keyframes gradientBG {
      0% {background-position: 0% 50%;}
      50% {background-position: 100% 50%;}
      100% {background-position: 0% 50%;}
    }
    .stApp {
      background-color: rgba(0, 0, 0, 0.6);
      padding: 20px;
      border-radius: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# ======================
# Backend Helpers
# ======================

def pitches_in_bar(track, bar_start, bar_end):
    pcs = []
    for n in track.notes:
        if n.time >= bar_start and n.time < bar_end:
            pcs.append(n.pitch % 12)
    return pcs

def detect_chord_from_pcs(pcs):
    if not pcs:
        return "N"
    counts = np.bincount(pcs, minlength=12)
    root = int(np.argmax(counts))
    has_major3 = ((root + 4) % 12) in pcs
    has_minor3 = ((root + 3) % 12) in pcs
    if has_major3:
        quality = "maj"
    elif has_minor3:
        quality = "min"
    else:
        quality = "other"
    return f"{root}:{quality}"

def midi_to_chord_sequence(path, bar_beats=4):
    music = muspy.read_midi(path)
    if not music.tracks:
        return []
    track = None
    for t in music.tracks:
        if t.notes:
            track = t
            break
    if track is None:
        return []
    res = music.resolution
    ticks_per_bar = res * bar_beats
    total_ticks = max((n.time+n.duration) for n in track.notes)
    seq, last_chord = [], None
    for start in range(0, total_ticks+1, ticks_per_bar):
        pcs = pitches_in_bar(track, start, start+ticks_per_bar)
        chord = detect_chord_from_pcs(pcs)
        if chord == "N" and last_chord is not None:
            chord = last_chord
        elif chord == "N":
            chord = "0:maj"
        seq.append(chord)
        last_chord = chord
    return seq

def build_vocab(corpus, min_count=1):
    flat = [c for seq in corpus for c in seq if c != "N"]
    counts = Counter(flat)
    chords = [c for c, cnt in counts.items() if cnt >= min_count]
    chords = sorted(chords)
    chords = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"] + chords
    stoi = {c:i for i,c in enumerate(chords)}
    itos = {i:c for c,i in stoi.items()}
    return stoi, itos

def sequences_to_indices(corpus, stoi, seq_len=16):
    X, Y = [], []
    for seq in corpus:
        idxs = [stoi.get(c, stoi["<UNK>"]) for c in seq]
        for i in range(len(idxs)-1):
            x = idxs[max(0, i-seq_len+1):i+1]
            if len(x) < seq_len:
                x = [stoi["<PAD>"]] * (seq_len - len(x)) + x
            y = idxs[i+1]
            X.append(x)
            Y.append(y)
    return X, Y

class ChordRNN(nn.Module):
    def __init__(self, vocab_size, emb=64, hidden=128, n_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hidden, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, x, hidden=None):
        e = self.embed(x)
        out, hidden = self.lstm(e, hidden)
        out = self.fc(out[:, -1, :])
        return out, hidden

def train_model(X, Y, vocab_size, epochs=20, batch_size=64, device="cpu"):
    X = torch.tensor(X, dtype=torch.long)
    Y = torch.tensor(Y, dtype=torch.long)
    ds = TensorDataset(X, Y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model = ChordRNN(vocab_size).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(epochs):
        total = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits, _ = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * xb.size(0)
        print(f"Epoch {ep+1}, loss={total/len(ds):.4f}")
    return model

def sample_chords(model, stoi, itos, length=16, temperature=1.0, device="cpu"):
    model.eval()
    seq_len = 16
    seq_input = [stoi["<PAD>"]] * (seq_len-1) + [stoi["<SOS>"]]
    generated = ["<SOS>"]
    hidden = None
    for _ in range(length):
        x = torch.tensor([seq_input], dtype=torch.long).to(device)
        logits, hidden = model(x, hidden)
        logits = logits.squeeze(0) / temperature
        probs = torch.softmax(logits, dim=-1).cpu().detach().numpy()
        if "N" in stoi:
            probs[stoi["N"]] = 0
            probs = probs / probs.sum()
        nxt = np.random.choice(len(probs), p=probs)
        chord = itos.get(nxt, "<UNK>")
        generated.append(chord)
        seq_input = seq_input[1:] + [nxt]
    return generated

def chord_token_to_pitches(token):
    if token in ("<PAD>","<SOS>","<EOS>","<UNK>"):
        return []
    try:
        root_str, quality = token.split(":")
        root = int(root_str)
    except:
        return []
    midi_root = 60 + root
    if quality == "maj":
        return [midi_root, midi_root+4, midi_root+7]
    elif quality == "min":
        return [midi_root, midi_root+3, midi_root+7]
    else:
        return [midi_root, midi_root+4, midi_root+7]

def chords_to_midi(chord_seq, filename="out.mid", ticks_per_quarter=24, beats_per_bar=4):
    music = muspy.Music(resolution=ticks_per_quarter)
    track = muspy.Track(program=0, is_drum=False, name="Chords")
    music.tracks.append(track)
    ticks_per_bar = ticks_per_quarter * beats_per_bar
    time = 0
    for chord in chord_seq:
        pitches = chord_token_to_pitches(chord)
        for p in pitches:
            track.notes.append(muspy.Note(time=time, pitch=p, duration=ticks_per_bar, velocity=80))
        time += ticks_per_bar
    muspy.write_midi(filename, music)

def midi_to_events(path):
    pm = pretty_midi.PrettyMIDI(path)
    events = []
    for inst in pm.instruments:
        for note in inst.notes:
            events.append({
                "type": "note",
                "pitch": note.pitch,
                "start": int(note.start * 480),
                "end": int(note.end * 480),
                "velocity": note.velocity
            })
    events = sorted(events, key=lambda x: x["start"])
    return events

# ======================
# Streamlit App
# ======================

st.title("🎶 TREBLE CHEF")
st.markdown("---")

prompt = st.text_area("Give Us A Few Words To Cook A Beat (Keywords for the lyrics, theme, etc.)", placeholder="e.g. Anything That Comes On Your Mind")

if st.button("Generate Music 🎵"):
    if prompt.strip() == "":
        st.warning("⚠️ Please enter something first!")
    else:
        # Load dataset of MIDIs
        files = glob.glob("./EDM Progressions/*/*.mid", recursive=True)
        corpus = [midi_to_chord_sequence(f) for f in files if midi_to_chord_sequence(f)]
        stoi, itos = build_vocab(corpus)
        X, Y = sequences_to_indices(corpus, stoi)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = train_model(X, Y, vocab_size=len(stoi), epochs=5, device=device)

        generated = sample_chords(model, stoi, itos, length=16)
        chords_to_midi(generated, "generated_chords.mid")

        # Convert to JSON for Gemini
        events = midi_to_events("generated_chords.mid")
        events_json = json.dumps(events, indent=2)
        print(events_json)

        client = genai.Client(api_key="AIzaSyD1s5Cw5rfV_daH0NC1mD3m05U8gWRvp1Q")
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f'Analyze the MIDI output: {events_json} but do not return the analysis and generate lyrics for a song (speed 120 bpm) with the keywords: {prompt}. Only return the lyrics with their corresponding chords for the entire duration of the song'
        )
        lyrics = response.text
        print(lyrics)

        # --- Display Lyrics ---
        st.subheader("Generated Lyrics")
        st.write(lyrics)

        # Convert MIDI to wav
        # Initialize with a SoundFont file
        fs = FluidSynth('FluidR3_GM.sf2')

        # Convert MIDI to WAV
        fs.midi_to_audio("generated_chords.mid", "output.wav")
        # --- Display Audio ---
        st.subheader("Generated Music")
        st.audio("output.wav", format="audio/wav")
