{/* Change directory of file_front and file_side to correct directories of frames
  (the frame number does not matter since it will be changed from houdini anyway depending on current frame)*/}
const directory = "Path\\to\\sketch2fluid"

{/* Do not change this part... */}
var file_front = File(directory + "\\houdini\\real-time_img\\front\\03.png");
var file_side = File(directory + "\\houdini\\real-time_img\\side\\03.png");
var refDoc_front = app.open(file_front);
var refDoc_side = app.open(file_side);
