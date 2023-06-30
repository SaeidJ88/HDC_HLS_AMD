Team Number:   xohw23-162
Project name: Real-Time Stream Data Processing System Using Hyperdimensional Computing and HLS 
Link to YouTube Video(s): https://youtu.be/y_HwIpJFFAw
Link to Project Repository: https://github.com/SaeidJ88/HDC_HLS_AMD.git
University Name: Sapienza di Roma
Participant(s): Saeid Jamili, Marco Angioli
Email(s): saeid.jamili@uniroma1.it, marco.angioli@uniroma1.it
Supervisor Name: Antonio Mastrandrea
Supervisor Email: antonio.mastrandrea@uniroma1.it
Board Used: Zybo 7020 
Software Version: Vivado 2023.1, Anaconda 2.4.0 (Python 3.10), MATLAB 2023a

Brief description of project:
This project presents a robust system for real-time stream data processing using Hyperdimensional Computing (HDC) and High-Level Synthesis (HLS). 
By leveraging the high-dimensional vectors in HDC and the efficient design methodology of HLS, the system can process stream data in real-time, 
demonstrating significant potential for edge computing and large-scale data processing applications.

Description of archive:
- `train`: Contains the Python script used for training the hyperdimensional vectors.
- `hls_vitis`: Houses the High-Level Synthesis (HLS) implementation files and testbench.
- `vitis`: Includes test projects with the PS (Processing System) part for the Vitis platform.
- `viv`: This directory contains the Vivado hardware project.
- `data_conversion`: Contains scripts used for converting and formatting the trained data.

Instructions to build and test project

1- Ensure Python 3.10, Vivado 2023.1, and MATLAB 2023a are properly installed on your system.
2- Clone the repository onto your local machine.
3- (Optional) If you want to modify the training code, you can do so by changing the Python code in the train directory and running the code.
4- (Optional) If you modified the model, you need to run the Matlab script in the data_conversion folder to extract the data that needs to be added to the HLS project.
5- To test the system, navigate to the hls directory and execute the HLS project. 
Run the provided simulation script to verify output results, and initiate the co-simulation script to examine waveforms and inference time.
6- (Optional) You can test the project on the board by running the Vitis project in the vitis directory.