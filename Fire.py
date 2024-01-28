import cv2
import pygame

pygame.mixer.init()
pygame.mixer.music.load('audio.mp3')

fire_cascade = cv2.CascadeClassifier('fire_detection.xml')
cap = cv2.VideoCapture(0)

sound_played = False

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fires = fire_cascade.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in fires:
        cv2.rectangle(frame, (x-20, y-20), (x+w+20, y+h+20), (255, 0, 0), 2)
        cv2.putText(frame, "Fire alert", (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if not sound_played:
            print("Fire is detected")
            pygame.mixer.music.play()
            sound_played = True

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
