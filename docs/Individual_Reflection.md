# Individual Internship Reflection

**Name:** Harsh Gupta  
**Internship Projects:** Autonomous Driving Perception Stack and TensorRT Model Optimization

## Overview

During my internship, I worked on developing an autonomous driving perception prototype and optimizing an object detection model for faster inference. These projects helped me understand how different AI and computer vision technologies can work together in a practical application.

## Technical Learning

I implemented object detection using YOLO11n and object tracking using ByteTrack. I also explored lane detection using OpenCV techniques such as Canny edge detection and Hough transforms, along with relative depth estimation using MiDaS.

For model optimization, I exported YOLO11n to ONNX and used NVIDIA TensorRT with FP16 precision on an RTX 4060 GPU. The benchmark showed a reduction in measured latency from 25.85 ms with PyTorch FP32 to 13.06 ms with TensorRT FP16, representing approximately a 1.98× speedup.

I also gained experience using GitHub for version control and Weights & Biases for recording experiment results.

## Challenges Faced

One challenge was integrating multiple computer vision techniques and understanding the limitations of their outputs. For example, traditional lane detection was sensitive to road markings, camera angles and other edges in the scene.

Setting up the GPU environment and exporting the model to TensorRT also required troubleshooting compatibility and dependency issues.

These experiences taught me the importance of testing individual components before combining them.

## Project Limitations

The completed work represents a perception and optimization prototype rather than a fully autonomous driving system. CARLA and ROS2 integration, additional model fine-tuning and real-time vehicle control remain future development goals.

## Movie Recommendation System: Individual Reflection

During this project, I learned how a two-stage recommendation system combines retrieval and ranking to generate personalized movie suggestions.

I worked with Python, TensorFlow, FAISS, and Streamlit to build and test the recommendation workflow. The retrieval stage identifies candidate movies using learned embeddings, while the ranking stage predicts ratings and sorts the results.

One challenge was managing the dependencies and saved model files required to run the application. I also learned how to organize project code, document the implementation, and present the results through a working demonstration.

This project improved my understanding of recommendation systems, machine learning pipelines, model inference, and practical application development.

## Conclusion

This internship strengthened my practical understanding of machine learning, computer vision, model deployment and performance optimization. More importantly, it helped me become more confident in debugging technical problems, interpreting benchmark results and documenting my work.

I hope to build on these skills by developing more advanced and reliable AI systems in the future.