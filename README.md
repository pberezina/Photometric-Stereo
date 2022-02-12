# Photometric-Stereo

<h1 align="center"><Photometric Stereo Algorithm></h1>

Given 3+ images of an object taken from different angles, the algorithm reconstructs 3D shape of that object object using the photometric stereo approach. Theory is overviewed in “Physics‐Based Vision: Principles and Practice, Shape Recovery” by Wolff, Shafer, Healey, and Peters (1992), and “Chapter 1 Photometric Stereo: An Overview” by Argyriou and Petrou (2009). 

In the project directory, you can run:

#### `python pstereo.py \Apple`
`pstereo.py` from the command line with an argument `input_path` set to the location of input images. Output images are automatically saved within that folder into "Results".


### Screenshots

<p float="left">
  <img src="./Pear/results/normals.png" width="210" title="Normal map" />
  <img src="./Pear/results/Albedo_gray.png" width="210" title="Albedo gray-scale map"/> 
  <img src="./Pear/results/color.png" width="210" title="Albedo RGB map"/>
  <img src="./Pear/results/rerendered.png" width="210" title="Re-rendered image"/>
</p>
<p></p>
<p float="left">
  <img src="./Elephant/results/normals.png" width="210" title="Normal map" />
  <img src="./Elephant/results/Albedo_gray.png" width="210" title="Albedo gray-scale map"/> 
  <img src="./Elephant/results/color.png" width="210" title="Albedo RGB map"/>
  <img src="./Elephant/results/rerendered.png" width="210" title="Re-rendered image"/>
</p>

### Built With

- Python
- CV2
- Numpy
- GDAL
