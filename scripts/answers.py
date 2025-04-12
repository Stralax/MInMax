import sys
import requests
import os

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <image_url>")
        return
    
    text= sys.argv[1]

    print(text)

if __name__ == "__main__":
    main()
