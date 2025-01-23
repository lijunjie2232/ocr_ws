from xg import Word
import pickle
from pathlib import Path

if __name__ == "__main__":
    ROOT = Path(__file__).parent
    DATA_DIR = ROOT / "xg_out"
    
    with open(DATA_DIR / "data.pickle", "rb") as f:
        word_list = pickle.load(f)
    
    wordtxt = DATA_DIR / "words.txt"
    linePattern = "%s,%s\n"
    with wordtxt.open("w") as f:
        for word in word_list:
            f.write(linePattern % (word.kanji, word.meaning))
            f.write(linePattern % (word.hina, word.meaning))
            
        