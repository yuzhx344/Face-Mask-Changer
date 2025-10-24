Face Mask Changer - Interactive Sichuan Opera Face-Changing Program

Project Overview

An interactive art program based on Python that combines traditional Sichuan Opera face-changing culture with modern computer vision technology. The program captures users' faces in real-time through the camera and overlays virtual masks when faces are detected. When users cover their faces with hands and then remove them, the masks automatically switch, simulating the magical effect of Sichuan Opera face-changing.

Inspiration

Collision of Tradition and Modernity:

· Inspired by Chinese intangible cultural heritage - Sichuan Opera face-changing art
· Combining traditional performing arts with digital media technology
· Exploring how to inherit and display traditional culture in new forms in the digital age

Interaction Design Philosophy:

· The mystery of "instant transformation" in face-changing art
· The naturalness and intuitiveness of gesture interaction
· Real-time feedback for immersive experience

Purpose and Significance

Educational Purpose

· Introduce traditional opera culture to younger generations through fun interactions
· Demonstrate the application of computer vision technology in cultural inheritance
· Provide practical cases combining programming learning and artistic creation

Technical Goals

· Implement real-time face detection and tracking
· Develop natural gesture interaction recognition system
· Create smooth visual feedback experience

Artistic Value

· Explore the integration of digital art and traditional culture
· Create personalized interactive art experiences
· Promote innovative development of traditional culture in the digital age

Features

Core Functions

· Real-time camera face detection
· Dynamic mask overlay display
· Gesture-triggered mask switching (occlusion detection)
· Multiple masks cycling transformation

User Experience

· Intuitive visual feedback
· Natural interaction methods
· Real-time status prompts
· Simple operation interface

Technical Implementation

Technology Stack

· OpenCV: Computer vision processing
· NumPy: Image data processing
· Haar Cascade: Face detection algorithm

System Architecture

```
Camera Input → Face Detection → Occlusion Judgment → Mask Switching → Visual Output
```

Installation and Operation

Environment Requirements

· Python 3.6+
· Camera device

Dependency Installation

```bash
pip install opencv-python numpy
```

Running the Program

```bash
python face_mask_changer.py
```

Usage Instructions

1. Ensure the camera works properly
2. Use in well-lit environment
3. Face the camera and maintain appropriate distance
4. Completely cover your face with hands for about 1 second, then remove to switch masks
5. Press 'q' to exit the program

Expansion Possibilities

Technical Expansion

· Integrate more accurate facial landmark detection
· Add AR special effects and animations
· Support custom mask uploads
· Add sound feedback and background music

Artistic Expansion

· Design richer mask library
· Add different opera style themes
· Achieve multi-person simultaneous face-changing effects
· Create plot-based interactive narratives


