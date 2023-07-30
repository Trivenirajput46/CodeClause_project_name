import os
import pygame

def get_valid_music_files(directory):
    valid_extensions = ['.mp3', '.wav', '.ogg']  # Add more if needed
    music_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1].lower() in valid_extensions:
                music_files.append(os.path.join(root, file))
    return music_files

def print_playlist(playlist):
    print("Available songs in the playlist:")
    for index, song in enumerate(playlist):
        print(f"{index + 1}. {os.path.basename(song)}")

def play_music(playlist, index):
    pygame.mixer.init()
    pygame.mixer.music.load(playlist[index])
    pygame.mixer.music.play()

def main():
    music_directory = "C:\\Users\Triveni Rajput\\Music"
    music_playlist = get_valid_music_files(music_directory)

    if not music_playlist:
        print("No valid music files found in the specified directory.")
        return

    print_playlist(music_playlist)

    while True:
        try:
            choice = int(input("Enter the song number to play (0 to exit): "))
            if choice == 0:
                print("Exiting the music player. Goodbye!")
                break
            elif 1 <= choice <= len(music_playlist):
                index_to_play = choice - 1
                play_music(music_playlist, index_to_play)
                while pygame.mixer.music.get_busy():
                    pass  # Wait for the song to finish playing
            else:
                print("Invalid input. Please enter a valid song number.")
        except ValueError:
            print("Invalid input. Please enter a valid song number.")

if __name__ == "__main__":
    main()
