**README.txt**

#Omertà - Morse Code Translator & Audio Encoder/Decoder

## Overview
Omertà is a feature-rich Morse code translator application designed with a sleek, modern GUI using CustomTkinter. It offers real-time text and Morse code conversion, audio playback with adjustable speed and frequency, waveform generation, file handling, and even advanced music encoding/decoding capabilities.

## Key Features
- **Real-Time Validation:** Automatically checks for valid input while typing.
- **Morse Code Conversion:** Convert text to Morse code and vice versa.
- **Audio Playback:** Play Morse code as audio with customizable speed and frequency.
- **Waveform Generation:** Create animated waveforms of Morse code and save them as images.
- **File Handling:** Export Morse code to text files or upload files for conversion.
- **Music Encoding/Decoding:** Encode Morse code into MIDI music files and decode them back into text.
- **Custom GUI:** Built with CustomTkinter for a clean and intuitive user experience.

## How to Run
1. Ensure you have Python 3.x installed on your system.
2. Install the required dependencies using the `requirements.txt` file.
3. Run the main script using:
   ```
   python <script_name>.py
   ```

## Dependencies
All necessary dependencies are listed in `requirements.txt`. Install them using:
```bash
pip install -r requirements.txt
```

## Usage
1. Enter text or Morse code in the provided input field.
2. Use the buttons to convert between text and Morse code.
3. Adjust speed and frequency sliders for audio playback.
4. Explore advanced features like waveform creation and music encoding through the side panel.

## License
This project is licensed under the MIT License.

---

**requirements.txt**

```
customtkinter
opencv-python
numpy
pillow
matplotlib
scikit-learn
scipy
music21

pywavelets
