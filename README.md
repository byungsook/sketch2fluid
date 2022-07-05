# Sketch2fluid Houdini Plugin

Houdini plugin for the code from the "sketch2fluid" paper.
 
## Getting started

### Installation

Get the houdini_python_server code by cloning the following repository:

``
git clone https://gitlab.ethz.ch/cglsim/houdini_python_server.git
``

Then install conda and create a conda environment, called houdini, with python 3.8 and numpy

```
conda create -n houdini python=3.8 numpy scipy pillow matplotlib cupy

conda activate houdini

conda install -c pytorch=1.7.1 torchvision cudatoolkit=10.2

pip3 install git+https://github.com/pvigier/perlin-numpy
```

### Start the Houdini plugin

Open the example.hipnc file in the houdini folder.

When opened the first time some directories might be missing. The directories can be changed in the sketch2fluid node in the sketch2fluid tab. After changing the directories press `Reload directories`. This will set the directories and open the connection shortly to test if everything works fine.

_(If Houdini should be able to open the sketches in Photoshop directly the location of the photoshop.exe file needs to be specified and the path to the sketch2fluid folder needs to be changed in the `openfile.jsx` file.)_

## Sketching

First the server/client connection needs to be established. The connection can be opened by pressing the `Open Connection` button and closed by the `Close Connection` button. When closed wait for the command promp to say "press any key to continue..." and it will close when pressed any key. See troubleshooting below if the connection does not open correctly.

There exist two modes for sketching:
1. `sketch mode (1)`: This mode is used to create a density from two sketches (front and left). The density is automatically created so changes made to the sketches will be shown immediately.

2. `refinement mode (2)`: In this mode changes made to the sketch will only refine the density while it tries to roughly keep the original shape.
    * Rotate density in Houdini which will create sketches from other sides. 
    * Make changes to the sketches by editing the sketches created by the differentiable sketcher.
    * Press `save refinement`. This will save the changes made to the density in the current position. 
      _**Important:** Don't rotate after refinement without saving, since this will delete the changes made in this position._
      
The sketches are saved in the folder `houdini\real-time_img\front` and `houdini\real-time_img\side`. They can be edited using any painting tool. If Photoshop was set up in Houdini pressing `open Photoshop` will open the sketches of the current frame.

### Sketching inside Houdini

If preferred it is possible to draw directly in Houdini on the planes next to the density using the Attribute Paint Nodes inside of the paint geometry node. Activating `automatic saving active` in the sketch2fluid node enables the density to automatically change while drawing on the planes.

_(Only use this in sketch or refinement mode and not in combination with another tool, because it might overwrite the sketches)_

## Super resolution

## Troubleshooting

* As long as the connection is open never close the command prompt that popped up when opening the connection. This will destroy the server/client connection.
If it happens close the connection, this will return an error like "No connection could be made because the target machine actively refused it". Ignore the error, reload the directories and open the connection again. This should open the connection normal again.

* If the same error happens due to another weird reason or the connection just stops working the same procedure normally works: close connection, reload directories, open connection again.

* If rotating in Houdini rotates the density differently than the sketches, the transform order in Houdini might be wrong. Make sure the order in the sketch2fluid node is `RzRxRy`. 

* If something happens to the density while refining it is almost always possible to transform back to the original position and create the density from sketches again by pressing `Redo volume from sketches`