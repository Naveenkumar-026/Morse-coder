import tkinter as tk
from tkinter import ttk 
import customtkinter as ctk
import winsound
import threading
import time
import tkinter.filedialog as fd
import math
import os
import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import pywt
from music21 import stream, note, chord, instrument, dynamics, converter, tempo
from tkinter import filedialog, END
import os


# Morse code dictionary
MORSE_CODE_DICT = {
    'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.',
    'F': '..-.', 'G': '--.', 'H': '....', 'I': '..', 'J': '.---',
    'K': '-.-', 'L': '.-..', 'M': '--', 'N': '-.', 'O': '---',
    'P': '.--.', 'Q': '--.-', 'R': '.-.', 'S': '...', 'T': '-',
    'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-', 'Y': '-.--',
    'Z': '--..', '1': '.----', '2': '..---', '3': '...--', '4': '....-', '5': '.....',
    '6': '-....', '7': '--...', '8': '---..', '9': '----.', '0': '-----', ' ': '/'
}

# Default settings for speed and frequency
unit_duration = 0.3  # Base unit duration (default 300ms)
frequency = 800  # Default frequency in Hz

# Function to translate text to Morse code
def text_to_morse(text):
    return ' '.join(MORSE_CODE_DICT.get(char.upper(), '') for char in text)

# Function to validate input for text-to-Morse conversion
def validate_text_input(text):
    valid_chars = set(MORSE_CODE_DICT.keys())  # Supported characters
    invalid_chars = [char for char in text.upper() if char not in valid_chars]
    return invalid_chars

# Function to validate input for Morse-to-text conversion
def validate_morse_input(morse_code):
    valid_chars = {'.', '-', '/', ' '}  # Supported Morse code characters
    invalid_chars = [char for char in morse_code if char not in valid_chars]
    return invalid_chars


# Function to handle real-time validation while typing(live)
def lreal_time_validation(event):
    input_text = linput_entry.get().strip()
    # Determine if the input is text or Morse based on allowed characters
    if all(char.upper() in MORSE_CODE_DICT.keys() or char == ' ' for char in input_text):
        linput_entry.configure(border_color="green")  # Valid text input
    elif all(char in ".-/ " for char in input_text):
        linput_entry.configure(border_color="green")  # Valid Morse code input
    else:
        linput_entry.configure(border_color="red")  # Invalid input


# Modified conversion functions with validation(live)
def lconvert_text_to_morse():
    text = linput_entry.get().strip()
    invalid_chars = validate_text_input(text)
    if invalid_chars:
        loutput_text.delete("1.0", tk.END)
        loutput_text.insert(tk.END, f"Error: Invalid characters: {', '.join(invalid_chars)}")
    else:
        morse_code = text_to_morse(text)
        loutput_text.delete("1.0", tk.END)
        loutput_text.insert(tk.END, morse_code)

def lconvert_morse_to_text():
    morse_code = linput_entry.get().strip()
    invalid_chars = validate_morse_input(morse_code)
    if invalid_chars:
        loutput_text.delete("1.0", tk.END)
        loutput_text.insert(tk.END, f"Error: Invalid characters: {', '.join(invalid_chars)}")
    else:
        words = morse_code.split(' / ')  # Split Morse code into words
        decoded_message = []
        for word in words:
            letters = word.split()  # Split words into letters
            decoded_message.append(''.join({v: k for k, v in MORSE_CODE_DICT.items()}.get(letter, '') for letter in letters))
        loutput_text.delete("1.0", tk.END)
        loutput_text.insert(tk.END, ' '.join(decoded_message))






# Function to handle real-time validation while typing
def real_time_validation(event):
    input_text = input_entry.get().strip()
    # Determine if the input is text or Morse based on allowed characters
    if all(char.upper() in MORSE_CODE_DICT.keys() or char == ' ' for char in input_text):
        input_entry.configure(border_color="green")  # Valid text input
    elif all(char in ".-/ " for char in input_text):
        input_entry.configure(border_color="green")  # Valid Morse code input
    else:
        input_entry.configure(border_color="red")  # Invalid input


# Modified conversion functions with validation
def convert_text_to_morse():
    text = input_entry.get().strip()
    invalid_chars = validate_text_input(text)
    if invalid_chars:
        output_text.delete("1.0", tk.END)
        output_text.insert(tk.END, f"Error: Invalid characters: {', '.join(invalid_chars)}")
    else:
        morse_code = text_to_morse(text)
        output_text.delete("1.0", tk.END)
        output_text.insert(tk.END, morse_code)

def convert_morse_to_text():
    morse_code = input_entry.get().strip()
    invalid_chars = validate_morse_input(morse_code)
    if invalid_chars:
        output_text.delete("1.0", tk.END)
        output_text.insert(tk.END, f"Error: Invalid characters: {', '.join(invalid_chars)}")
    else:
        words = morse_code.split(' / ')  # Split Morse code into words
        decoded_message = []
        for word in words:
            letters = word.split()  # Split words into letters
            decoded_message.append(''.join({v: k for k, v in MORSE_CODE_DICT.items()}.get(letter, '') for letter in letters))
        output_text.delete("1.0", tk.END)
        output_text.insert(tk.END, ' '.join(decoded_message))



        

# Function to play and display Morse code with color coding
def play_and_display_morse_code(morse_code):
    loutput_text.delete("1.0", tk.END)  # Clear previous output

    for char in morse_code:
        if char == '.':
            winsound.Beep(frequency, int(unit_duration * 1000))  # Dot: 1 unit
            loutput_text.insert(tk.END, '.', "dot")  # Insert dot with color
        elif char == '-':
            winsound.Beep(frequency, int(unit_duration * 3 * 1000))  # Dash: 3 units
            loutput_text.insert(tk.END, '-', "dash")  # Insert dash with color
        elif char == ' ':
            time.sleep(unit_duration * 3)  # Gap between letters
            loutput_text.insert(tk.END, ' ')  # Insert space
        elif char == '/':
            time.sleep(unit_duration * 7)  # Gap between words
            loutput_text.insert(tk.END, ' / ')  # Insert word separator

        # Update the text box for real-time display
        loutput_text.update()
        time.sleep(unit_duration * 0.5)  # Pause between symbols for synchronization

def play_audio_and_generate_morse():
    morse_code = text_to_morse(linput_entry.get().strip())  # Generate Morse code
    if morse_code:  # Only proceed if there is valid Morse code
        # Start a new thread for simultaneous audio and text generation
        threading.Thread(target=play_and_display_morse_code, args=(morse_code,), daemon=True).start()

# Functions to update speed and frequency
def update_speed(value):
    global unit_duration
    unit_duration = float(value) / 1000  # Convert milliseconds to seconds
    speed_value_label.configure(text=f"{int(value)} ms")  # Update speed label

def update_frequency(value):
    global frequency
    frequency = int(value)
    frequency_value_label.configure(text=f"{int(value)} Hz")  # Update frequency label

# Function to export the output to a text file
def export_to_file():
    # Get the content from the output_text box
    content = output_text.get("1.0", tk.END).strip()
    if content:  # Ensure there's something to export
        # Open a save file dialog
        file_path = fd.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if file_path:  # If the user selects a file path
            with open(file_path, "w") as file:
                file.write(content)
                print(f"Exported to {file_path}")  # Optional: Console feedback

# Function to handle file upload and translation
def upload_and_translate():
    # Open a file dialog to select a text file
    file_path = fd.askopenfilename(
        filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
    )
    if file_path:  # Proceed if a file is selected
        with open(file_path, "r") as file:
            content = file.read().strip()  # Read the file content

        # Determine if the content is Morse code or text
        if all(char in ".-/ " for char in content):  # Check for Morse code characters
            # Translate Morse code to text
            words = content.split(' / ')  # Split Morse code into words
            decoded_message = []
            for word in words:
                letters = word.split()  # Split words into letters
                decoded_message.append(''.join({v: k for k, v in MORSE_CODE_DICT.items()}.get(letter, '') for letter in letters))
            result = ' '.join(decoded_message)
        else:
            # Assume content is plain text and translate to Morse code
            result = text_to_morse(content)

        # Display the result in the output_text box
        output_text.delete("1.0", tk.END)
        output_text.insert(tk.END, result)

# Function to open waveform animation in a pop-up window
def create_waveform(morse_code):
    canvas_width = max(600, len(morse_code) * 50)  # Dynamic width based on Morse code length
    canvas_height = 200
    waveform_window = tk.Toplevel()
    waveform_window.title("Waveform Animation")
    waveform_window.geometry(f"{canvas_width}x{canvas_height + 50}")

    waveform_canvas = tk.Canvas(waveform_window, width=canvas_width, height=canvas_height, bg="black")
    waveform_canvas.pack(pady=10)

    base_y = canvas_height // 2
    x = 10
    dot_amplitude = canvas_height * 0.15
    dash_amplitude = canvas_height * 0.3
    max_code_length = max(1, len(morse_code))
    unit_width = max(canvas_width // (max_code_length * 6), 5)

    def draw_wave(amplitude, wavelength, color):
        nonlocal x
        points = []
        for i in range(wavelength):
            x1 = x + i
            y1 = base_y - amplitude * math.sin(2 * math.pi * i / wavelength)
            points.append((x1, y1))
        for i in range(1, len(points)):
            waveform_canvas.create_line(points[i - 1][0], points[i - 1][1], points[i][0], points[i][1], fill=color, width=2)
        x += wavelength + unit_width // 2  # Add extra space after each waveform

    def save_waveform():
        file_path = fd.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Files", "*.png"), ("All Files", "*.*")],
        )
        if file_path:
            waveform_canvas.postscript(file=f"{file_path}.eps", colormode="color")
            img = Image.open(f"{file_path}.eps")
            img = img.convert("RGBA")
            img.save(file_path, "png")
            img.close()

    save_button = tk.Button(waveform_window, text="Save Waveform", command=save_waveform)
    save_button.pack(pady=10)

    progress = ttk.Progressbar(waveform_window, orient="horizontal", length=canvas_width, mode="determinate")
    progress.pack(pady=10)

    def animate_waveform(index=0):
        nonlocal x
        if index >= len(morse_code):
            progress["value"] = 100
            return

        progress["value"] = (index / len(morse_code)) * 100
        char = morse_code[index]
        if char == '.':
            draw_wave(dot_amplitude, unit_width, "green")
            x += unit_width // 2
        elif char == '-':
            draw_wave(dash_amplitude, unit_width * 2, "blue")
            x += unit_width
        elif char == ' ':
            x += unit_width * 3  # Space between letters
        elif char == '/':
            x += unit_width * 6  # Space between words

        delay = max(10, 2000 // len(morse_code))  # Adjust delay for clarity
        waveform_window.after(delay, animate_waveform, index + 1)

    animate_waveform()

# Function to process the waveform image
def process_waveform_image():
    try:
        # Open file dialog to select an image
        file_path = fd.askopenfilename(
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp"), ("All Files", "*.*")]
        )
        if not file_path:
            return

        # Load the image
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        # Convert to binary image
        _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

        # Detect contours
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours based on height and width
        filtered_contours = [
            c for c in contours if 5 < cv2.boundingRect(c)[2] < 100 and 5 < cv2.boundingRect(c)[3] < 100
        ]

        # Sort contours by x-coordinate
        filtered_contours = sorted(filtered_contours, key=lambda c: cv2.boundingRect(c)[0])

        # Extract widths and gaps
        contour_widths = [cv2.boundingRect(c)[2] for c in filtered_contours]
        contour_x_coords = [cv2.boundingRect(c)[0] for c in filtered_contours]
        gaps = [contour_x_coords[i + 1] - (contour_x_coords[i] + contour_widths[i])
                for i in range(len(contour_x_coords) - 1)]

        # Normalize widths and gaps
        min_width, max_width = min(contour_widths), max(contour_widths)
        normalized_widths = [
            (w - min_width) / (max_width - min_width + 1e-5) for w in contour_widths
        ]
        min_gap, max_gap = min(gaps), max(gaps)
        normalized_gaps = [
            (g - min_gap) / (max_gap - min_gap + 1e-5) for g in gaps
        ]

        # Adjust thresholds
        dot_threshold_raw = min_width + (max_width - min_width) * 0.33
        dash_threshold_raw = min_width + (max_width - min_width) * 0.66

        print(f"Contour widths: {contour_widths}")
        print(f"Gaps: {gaps}")
        print(f"Normalized widths: {normalized_widths}")
        print(f"Normalized gaps: {normalized_gaps}")
        print(f"Dot threshold (raw widths): {dot_threshold_raw}")
        print(f"Dash threshold (raw widths): {dash_threshold_raw}")

        # Decode Morse code
        morse_code = ""
        letter_gap_threshold = 0.2
        word_gap_threshold = 0.8

        for i, width in enumerate(contour_widths):
            if width <= dot_threshold_raw:  # Dot
                morse_code += "."
                print(f"Width {width} classified as dot (Index: {i})")
            elif width >= dash_threshold_raw:  # Dash
                morse_code += "-"
                print(f"Width {width} classified as dash (Index: {i})")
            else:  # Ambiguous case
                # Classify closer to dot or dash
                if abs(width - dot_threshold_raw) < abs(width - dash_threshold_raw):
                    morse_code += "."
                    print(f"Width {width} classified as closer to dot (Index: {i})")
                else:
                    morse_code += "-"
                    print(f"Width {width} classified as closer to dash (Index: {i})")

            if i < len(normalized_gaps):  # Check gaps
                gap = normalized_gaps[i]
                if gap >= word_gap_threshold:  # Word gap
                    morse_code += " / "
                elif gap >= letter_gap_threshold:  # Letter gap
                    morse_code += " "

        print(f"Morse Code: {morse_code}")

        # Convert Morse code to text
        try:
            # Split Morse code into words
            words = morse_code.strip().split(" / ")
            decoded_message = []

            # Create an inverted dictionary for decoding
            inverted_dict = {v: k for k, v in MORSE_CODE_DICT.items()}

            for word in words:
                # Split each word into letters
                letters = word.strip().split()
                decoded_word = "".join(
                    inverted_dict.get(letter, f"[Unknown: {letter}]") for letter in letters
                )
                decoded_message.append(decoded_word)

            # Combine decoded words into a sentence
            decoded_text = " ".join(decoded_message)

            # Display the results
            output_text.delete("1.0", tk.END)
            output_text.insert(tk.END, f"Morse Code: {morse_code}\nDecoded Text: {decoded_text}")

        except Exception as e:
            print(f"Error decoding Morse code: {e}")
            

    except Exception as e:
        # Display any errors
        output_text.delete("1.0", tk.END)
        output_text.insert(tk.END, f"Error: {e}")


# Genre Configurations
GENRES = {
    'Piano': {'instrument': instrument.Piano(), 'dot_pitch': 'C4', 'dash_pitch': 'G4'},
    
    'Guitar': {'instrument': instrument.ElectricGuitar(), 'dot_pitch': 'A4', 'dash_pitch': 'F4'},
    
}

# Encode Morse Code with Enhancements
def mencode_morse(text, genre):
    genre_config = GENRES[genre]
    s = stream.Stream()
    s.append(genre_config['instrument'])
    
    crescendo = dynamics.Dynamic('mf')
    s.append(crescendo)
    
    morse_code = ' '.join(MORSE_CODE_DICT[char] for char in text.upper() if char in MORSE_CODE_DICT)
    
    for char in morse_code:
        if char == '.':
            s.append(note.Note(genre_config['dot_pitch'], quarterLength=0.25))
            s.append(chord.Chord([genre_config['dot_pitch'], 'E5'], quarterLength=0.25))
        elif char == '-':
            long_note = note.Note(genre_config['dash_pitch'], quarterLength=0.5)
            s.append(long_note)
            s.append(chord.Chord([genre_config['dash_pitch'], 'C4'], quarterLength=0.5))
        elif char == ' ':
            s.append(note.Rest(quarterLength=0.5))
    return s

# Decode Morse Code
def mdecode_morse(midi_file, genre):
    genre_config = GENRES[genre]
    morse_code = ""
    s = converter.parse(midi_file)
    
    for elem in s.flatten().notesAndRests:
        if isinstance(elem, note.Note):
            if elem.nameWithOctave == genre_config['dot_pitch']:
                morse_code += '.'
            elif elem.nameWithOctave == genre_config['dash_pitch']:
                morse_code += '-'
        elif isinstance(elem, note.Rest):
            morse_code += ' '
    return morse_code

# Translate Morse Code to Text
def mtranslate_morse_to_text(morse_code):
    reverse_dict = {v: k for k, v in MORSE_CODE_DICT.items()}
    words = morse_code.split('   ')
    decoded_text = ' '.join(''.join(reverse_dict[char] for char in word.split() if char in reverse_dict) for word in words)
    return decoded_text

# GUI Implementation

def mencode():
    text = minput_text.get()
    genre = genre_var.get()
    if text and genre:
        s = mencode_morse(text, genre)
        file_path = filedialog.asksaveasfilename(defaultextension=".mid", filetypes=[("MIDI files", "*.mid")])
        if file_path:
            s.write('midi', fp=file_path)
            mresult_display.insert(END, f"Encoded to MIDI: {file_path}\n")
    else:
        mresult_display.insert(END, "Error: Please enter text and select a genre.\n")

def mdecode():
    genre = genre_var.get()
    file_path = filedialog.askopenfilename(filetypes=[("MIDI files", "*.mid")])
    if file_path and genre:
        decoded_text = mdecode_morse(file_path, genre)
        translated_text = mtranslate_morse_to_text(decoded_text)
        mresult_display.insert(END, f"Decoded Morse Code: {decoded_text}\nTranslated Text: {translated_text}\n")
    else:
        mresult_display.insert(END, "Error: Please select a genre and a valid MIDI file.\n")

def mplay_midi():
    file_path = filedialog.askopenfilename(filetypes=[("MIDI files", "*.mid")])
    if file_path:
        os.system(f'start wmplayer "{file_path}"')
        mresult_display.insert(END, f"Playing MIDI: {file_path}\n")
    else:
        mresult_display.insert(END, "Error: No MIDI file selected.\n")

        
# Initialize CustomTkinter
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Create the main window
root = ctk.CTk()
root.title("4by Omertà")
root.geometry("600x900")


# Notebook (Tabs)
notebook = ctk.CTkTabview(root, width=500, height=800)
notebook.pack(padx=10, pady=10, expand=True, fill="both")

live = notebook.add("Straight Talk")
vault = notebook.add("Stash & Decode")


# UI Elements
title_label = ctk.CTkLabel(live, text="The Syndicate's Signal", font=("Arial", 20, "bold"))
title_label.pack(pady=10)

#for live
linput_label = ctk.CTkLabel(live, text="Enter Text:")
linput_label.pack(pady=5)

linput_entry = ctk.CTkEntry(live, width=400)
linput_entry.pack(pady=5)

# Bind the input entry to validate input in real-time
linput_entry.bind("<KeyRelease>", lreal_time_validation)

loutput_label = ctk.CTkLabel(live, text="Generated Morse Code:")
loutput_label.pack(pady=5)

# Replace CTkTextbox with tk.Text for styling support
loutput_text = ctk.CTkTextbox(live, height=200, width=400, font=("Arial", 12))
loutput_text.pack(pady=5)


#for vault
input_label = ctk.CTkLabel(vault, text="Enter Text:")
input_label.pack(pady=5)

input_entry = ctk.CTkEntry(vault, width=400)
input_entry.pack(pady=5)

# Bind the input entry to validate input in real-time
input_entry.bind("<KeyRelease>", real_time_validation)

output_label = ctk.CTkLabel(vault, text="Generated Morse Code:")
output_label.pack(pady=5)

# Replace CTkTextbox with tk.Text for styling support
output_text = ctk.CTkTextbox(vault, height=200, width=400, font=("Arial", 12))
output_text.pack(pady=5)


# Frame to hold the buttons side by side
button_frame = ctk.CTkFrame(live)
button_frame.pack(pady=10)

# Buttons for conversion
convert_to_morse_button = ctk.CTkButton(button_frame, text="Text to Morse", command=lconvert_text_to_morse)
convert_to_morse_button.pack(side="left", padx=5)

convert_to_text_button = ctk.CTkButton(button_frame, text="Morse to Text", command=lconvert_morse_to_text)
convert_to_text_button.pack(side="right", padx=5)

# Frame to hold the buttons side by side
vbutton_frame = ctk.CTkFrame(vault)
vbutton_frame.pack(pady=10)

# Buttons for conversion
convert_to_morse_button = ctk.CTkButton(vbutton_frame, text="Text to Morse", command=convert_text_to_morse)
convert_to_morse_button.pack(side="left", padx=5)

convert_to_text_button = ctk.CTkButton(vbutton_frame, text="Morse to Text", command=convert_morse_to_text)
convert_to_text_button.pack(side="right", padx=5)

# Speed Slider
speed_label = ctk.CTkLabel(live, text="Adjust Speed (ms per unit):")
speed_label.pack(pady=5)

speed_slider = ctk.CTkSlider(live, from_=300, to=1000, number_of_steps=70, command=update_speed)
speed_slider.set(300)  # Default speed (300ms)
speed_slider.pack(pady=5)

speed_value_label = ctk.CTkLabel(live, text="300 ms")  # Display current speed value
speed_value_label.pack(pady=5)

# Frequency Slider
frequency_label = ctk.CTkLabel(live, text="Adjust Frequency (Hz):")
frequency_label.pack(pady=5)

frequency_slider = ctk.CTkSlider(live, from_=400, to=1200, number_of_steps=80, command=update_frequency)
frequency_slider.set(800)  # Default frequency (800 Hz)
frequency_slider.pack(pady=5)

frequency_value_label = ctk.CTkLabel(live, text="800 Hz")  # Display current frequency value
frequency_value_label.pack(pady=5)

# Buttons
translate_to_morse_button = ctk.CTkButton(live, text="Play Morse Code", command=play_audio_and_generate_morse)
translate_to_morse_button.pack(pady=10)


# Frame to hold the buttons side by side
tt_frame = ctk.CTkFrame(vault)
tt_frame.pack(pady=10)

# Button for exporting the output to a file
export_button = ctk.CTkButton(tt_frame, text="Export to File", command=export_to_file)
export_button.pack(side="left", padx=5)

# Button for uploading a file and translating its content
upload_button = ctk.CTkButton(tt_frame, text="Upload and Translate", command=upload_and_translate)
upload_button.pack(side="right", padx=5)


# Frame to hold the buttons side by side
wt_frame = ctk.CTkFrame(vault)
wt_frame.pack(pady=10)

#Waveform window
waveform_button = ctk.CTkButton(wt_frame, text="Create Waveform", command=lambda: threading.Thread(target=create_waveform, args=(text_to_morse(input_entry.get().strip()),), daemon=True).start())
waveform_button.pack(side="left", padx=5)

# Button to upload and process waveform image
process_waveform_button = ctk.CTkButton(wt_frame, text="Upload Waveform", command=process_waveform_image)
process_waveform_button.pack(side="right", padx=5)



#Side bar

toggle_button = None  # Placeholder for the toggle button
side_window = None  # Placeholder for the external side window
side_window_visible = False  # Track visibility state





# Function to open the side panel
def open_side_panel():
    global side_window, side_window_visible, minput_text, mresult_display, genre_var

    # Create a new external window
    side_window = ctk.CTkToplevel(root)
    side_window.geometry("700x700")
    side_window.title("Rhythm Cartel")

    # Align the side panel to the right of the main window
    x_position = root.winfo_x() + root.winfo_width()
    y_position = root.winfo_y()
    side_window.geometry(f"+{x_position}+{y_position}")

    # Prevent resizing of the side panel
    side_window.resizable(False, False)


    # Input Fields
    minput_frame = ctk.CTkFrame(side_window)
    minput_frame.pack(pady=20, padx=20)
    
    ctk.CTkLabel(minput_frame, text="Enter Text to Encode:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
    minput_text = ctk.CTkEntry(minput_frame, width=300)
    minput_text.grid(row=0, column=1, padx=10, pady=10)

    ctk.CTkLabel(minput_frame, text="Select Instrument:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
    genre_var = ctk.StringVar(value="Select")
    genre_menu = ctk.CTkOptionMenu(minput_frame, variable=genre_var, values=list(GENRES.keys()))
    genre_menu.grid(row=1, column=1, padx=10, pady=10)

    # Buttons
    mbutton_frame = ctk.CTkFrame(side_window)
    mbutton_frame.pack(pady=10)

    mencode_button = ctk.CTkButton(mbutton_frame, text="Encode to Music", command=mencode)
    mencode_button.grid(row=0, column=0, padx=10, pady=10)

    mdecode_button = ctk.CTkButton(mbutton_frame, text="Decode from Music", command=mdecode)
    mdecode_button.grid(row=0, column=1, padx=10, pady=10)

    mplay_button = ctk.CTkButton(mbutton_frame, text="Play MIDI", command=mplay_midi)
    mplay_button.grid(row=0, column=2, padx=10, pady=10)

    # Result Display
    mresult_frame = ctk.CTkFrame(side_window)
    mresult_frame.pack(pady=20, padx=20)
    
    ctk.CTkLabel(mresult_frame, text="Results:").grid(row=0, column=0, padx=10, pady=10, sticky="nw")
    mresult_display = ctk.CTkTextbox(mresult_frame, width=400, height=200)
    mresult_display.grid(row=0, column=1, padx=10, pady=10)

    side_window_visible = True

# Function to toggle the external side panel
def toggle_side_panel():
    global side_window, side_window_visible

    if side_window_visible:
        # If side panel is open, close it
        side_window.destroy()
        side_window = None
        side_window_visible = False
    else:
        # If side panel is closed, open it
        open_side_panel()


# Add a button to toggle the side panel
toggle_button = ctk.CTkButton(vault, text="Tune & Decode", command=toggle_side_panel)
toggle_button.pack(pady=20)


# Run the application
root.mainloop()
