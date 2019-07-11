PARAMS = {
	# The UCF-101 dataset has 101 classes
	"num_classes": 101,

	# Images are cropped to (CROP_SIZE, CROP_SIZE)
	"crop_size": 112,
	"channels": 3,

	# Number of frames per video clip
	"num_frames_per_clip": 16,

	# batch size
	"batch_size": 32,

	# weight reg
	"reg_w": 0.0001,
}
