import streamlit as st
import chess
import chess.pgn
import chess.engine
import numpy as np
import tensorflow as tf
import io
import os

import os  # Make sure this is imported at the top

import os

@st.cache_resource
def load_model():
    # 1. Get the current directory where app.py is running
    current_dir = os.getcwd()
    st.write(f"📂 Current Working Directory: {current_dir}")
    
    # 2. List ALL files in this directory (so we can see if the model is there)
    files_in_dir = os.listdir(current_dir)
    st.write("📄 Files found here:", files_in_dir)
    
    # 3. Define the model path
    model_filename = 'trap_model.h5'
    
    # 4. Check if it exists
    if model_filename not in files_in_dir:
        st.error(f"❌ I cannot find '{model_filename}' in this folder.")
        st.error("Please make sure trap_model.h5 is uploaded to the same folder as app.py on GitHub.")
        return None
        
    # 5. Load it
    try:
        model = tf.keras.models.load_model(model_filename)
        st.success("✅ Model Loaded Successfully!")
        return model
    except Exception as e:
        st.error(f"❌ File found, but failed to load. Is it a valid .h5 file? Error: {e}")
        return None

# --- PAGE CONFIG ---
st.set_page_config(page_title="Chess Trap Detector", layout="wide")

st.title("♟️ The Chess Trap Detector")
st.write("Paste your game PGN below. This AI detects if you missed a 'Trap' — a move that looks safe but is actually a blunder.")

# --- LOAD RESOURCES ---
@st.cache_resource
def load_model():
    # Load the trained model
    return tf.keras.models.load_model('trap_model.h5')

@st.cache_resource
def load_engine():
    # Note: Stockfish setup on Streamlit Cloud is tricky. 
    # Usually, we need a static binary. For now, we will try to detect generic installation
    # OR you might need to upload a stockfish binary to your repo.
    # For this simple demo, we will try to use a basic check or just the AI model.
    return None 

model = load_model()

# --- HELPER: FEN TO MATRIX ---
def fen_to_matrix(fen):
    board = chess.Board(fen)
    matrix = np.zeros((8, 8, 12), dtype=int)
    piece_map = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = 7 - (square // 8)
            col = square % 8
            layer = piece_map[piece.symbol()]
            matrix[row, col, layer] = 1
    return matrix

# --- INPUT SECTION ---
pgn_input = st.text_area("Paste PGN Text Here:", height=200)

if st.button("Analyze Game"):
    if pgn_input:
        pgn_io = io.StringIO(pgn_input)
        game = chess.pgn.read_game(pgn_io)
        
        if game:
            board = game.board()
            st.success(f"Game Loaded: {game.headers.get('White')} vs {game.headers.get('Black')}")
            
            # Iterate through moves
            move_count = 1
            for move in game.mainline_moves():
                board.push(move)
                fen = board.fen()
                
                # PREDICT WITH AI
                # Convert current board to matrix
                input_matrix = fen_to_matrix(fen)
                # Reshape for the model (1, 8, 8, 12)
                input_matrix = np.expand_dims(input_matrix, axis=0)
                
                # Get prediction (0 to 1)
                prediction = model.predict(input_matrix, verbose=0)[0][0]
                
                # If prediction is high (> 0.5), it's a "Trap" scenario
                # You can adjust this threshold
                if prediction > 0.8: 
                    st.markdown(f"---")
                    st.warning(f"⚠️ **Trap Opportunity Detected at Move {move_count}**")
                    st.image(f"https://fen-to-image.com/image/{fen}", width=300)
                    st.write(f"**AI Confidence:** {round(prediction * 100, 1)}% that this position contains a tricky trap.")
                
                move_count += 1
        else:
            st.error("Invalid PGN. Please check your text.")
