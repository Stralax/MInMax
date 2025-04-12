import sys
import requests
import os

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <image_url>")
        return
    
    image_url = sys.argv[1]

    # Tukaj lahko dodaš logiko za obdelavo slike
    print("Hello")  # Samo izpiše "Hello", kot si želel

if __name__ == "__main__":
    main()
