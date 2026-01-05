# ComputerVision
Facedetection :
Tracks 11 key facial points to create a unique spatial profile.Normalization: Uses face-width ratios so verification works even if you move away from the camera.Custom Verification: Implements a Mean Absolute Error (MAE) threshold to compare live faces against the stored database.This is the most advanced computer vision project i have made till now
Built with Tkinter for seamless User Enrollment and Identity Verification
Uses a local JSON-based storage (face_db.json) to act as a lightweight user registry.(This is for small scale but it can be easily updated later on)
Optimisation and how it works:
I isolated 11 specific landmarks (Nose, Eyes, Lips, Jawline) from the 468 provided by MediaPipe.(If i processed all of them my pc was lagging)
The system generates every possible pair between these points to create a comprehensive spatial mesh.
To solve the problem of users being at different distances from the camera, I implemented Feature Scaling.Every Euclidean distance between point pairs is divided by the Face Width
Identity is confirmed by calculating the Mean Absolute Error (MAE) between the live embedding and the stored database vector


AI proctored Object detection:
A real-time monitoring tool using YOLOv8 to detect prohibited items like cell phones and calculators during exams
Optimsation used:Implemented a frame-skipping logic that only runs the heavy YOLO model every 2nd frame, significantly improving real-time FPS on standard hardware.
Filtered the COCO dataset to track only specific objects (phones, calculators, clocks) with a confidence threshold of >0.3 to reduce false positives.

Annoying Volume control:
This is a fun project honestly.There was a online trend where you need to make the most annoying volume controller you can it basically changes volume by judging how much your mouth is open

Want to listen to music at 100% volume? Hope you enjoy keeping your mouth wide open like you're at the dentist.
If you look away from the camera for a second, the system loses you and stops updating, leaving you stuck with whatever volume you last screamed at.


 How to Run(All the projects)
   ```bash
   pip install -r requirements.txt

   and then run the python file
